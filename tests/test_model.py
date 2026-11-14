"""Tests DeiT-Tiny mathematics, initialization, and pretrained weight loading.

Purpose:
    Keep model calculations, weight loading, and trainable parameters consistent with
    the study.

Tests:
    test_architecture_and_scratch_initialization:
        Catches changed model dimensions, scratch initialization, token inputs, or
        accepted image shapes.

    test_attention_matches_explicit_three_head_mathematics:
        Detects incorrect attention head splitting, scaling, mixing, or output
        projection.

    test_forward_matches_explicit_fp64_equations:
        Detects forward-pass calculations that disagree with explicit transformer
        equations.

    test_frozen_scope_gradients_and_optimizer_updates:
        Detects backbone updates or a head that fails to update during frozen training.

    test_complete_mapping_and_fresh_head:
        Catches incorrectly mapped pretrained weights or an overwritten nine-class head.

    test_incompatible_and_incomplete_weights_are_rejected:
        Detects acceptance of missing, unexpected, or incompatible pretrained tensors.

    test_pretrained_backbone_and_initializers:
        Detects real publisher weights loading incorrectly, replacing the head, or
        freezing the wrong parameters.
"""

import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from fairage.model import DeiTTiny, load_pretrained_weights
from fairage.model.deit import Attention
from fairage.train import configure_model


PRETRAINED_WEIGHTS = Path("data/weights/deit_tiny_imagenet1k/pytorch_model.bin")


def backbone_mapping():
    """List the expected ImageNet names independently of the loader's rewrite."""

    mapping = {"cls_token": "cls_token", "pos_embed": "pos_embed"}
    for suffix in ("weight", "bias"):
        mapping[f"patch_embed.{suffix}"] = f"patch_embed.proj.{suffix}"
        mapping[f"norm.{suffix}"] = f"norm.{suffix}"
        for index in range(12):
            for layer in ("norm1", "norm2", "attn.qkv", "attn.proj", "fc1", "fc2"):
                source = f"mlp.{layer}" if layer in ("fc1", "fc2") else layer
                mapping[f"blocks.{index}.{layer}.{suffix}"] = f"blocks.{index}.{source}.{suffix}"
    return mapping


class ModelMathChecks(unittest.TestCase):

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4531)

    def test_architecture_and_scratch_initialization(self):
        model = configure_model({"experiment": "scratch", "dropout": 0.0}).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5_526_153)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(model.patch_embed.in_channels, 3)
        self.assertEqual(model.patch_embed.out_channels, 192)
        self.assertEqual(model.patch_embed.kernel_size, (16, 16))
        self.assertEqual(model.patch_embed.stride, (16, 16))
        self.assertEqual(tuple(model.cls_token.shape), (1, 1, 192))
        self.assertEqual(tuple(model.pos_embed.shape), (1, 197, 192))
        self.assertEqual(len(model.blocks), 12)
        # Sampling tolerances check the declared initialization scales, rather
        # than requiring the same random draws across PyTorch versions.
        self.assertAlmostEqual(model.head.weight.std().item(), .02, delta=.002)
        self.assertAlmostEqual(model.pos_embed.std().item(), .02, delta=.001)
        self.assertAlmostEqual(model.cls_token.std().item(), 1e-6, delta=3e-7)
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                self.assertEqual(torch.count_nonzero(layer.bias).item(), 0)
        for block in model.blocks:
            self.assertEqual(block.attn.qkv.in_features, 192)
            self.assertEqual(block.attn.qkv.out_features, 3 * 3 * 64)
            self.assertEqual(block.fc1.out_features, 768)
            self.assertEqual(block.norm1.eps, 1e-6)
        images = torch.zeros(1, 3, 224, 224)
        first_tokens = []
        handle = model.blocks[0].register_forward_pre_hook(
            lambda module, inputs: first_tokens.append(inputs[0].detach().clone()))
        try:
            with torch.no_grad():
                patches = model.patch_embed(images).flatten(2).transpose(1, 2)
                features = model.forward_features(images)
                logits = model.head(features[:, 0])
                torch.testing.assert_close(model(images), logits, atol=0, rtol=0)
        finally:
            handle.remove()
        self.assertEqual(tuple(patches.shape), (1, 196, 192))
        self.assertEqual(tuple(features.shape), (1, 197, 192))
        self.assertEqual(tuple(logits.shape), (1, 9))
        torch.testing.assert_close(first_tokens[0][:, :1],
                                   model.cls_token + model.pos_embed[:, :1])
        torch.testing.assert_close(first_tokens[0][:, 1:], patches + model.pos_embed[:, 1:])
        for shape in ((1, 1, 224, 224), (1, 3, 225, 224), (3, 224, 224)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "Expected RGB"):
                model(torch.zeros(shape))

    def test_attention_matches_explicit_three_head_mathematics(self):
        attention = Attention(dropout=0.2).eval()
        tokens = torch.linspace(-0.8, 0.9, 2 * 5 * 192).reshape(2, 5, 192)
        with torch.no_grad():
            projections = F.linear(tokens, attention.qkv.weight, attention.qkv.bias)
            query, key, value = [part.reshape(2, 5, 3, 64).transpose(1, 2)
                                 for part in projections.chunk(3, dim=-1)]
            probabilities = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(64), dim=-1)
            explicit = probabilities @ value
            optimized = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
            torch.testing.assert_close(optimized, explicit, atol=1e-6, rtol=1e-5)
            expected = F.linear(explicit.transpose(1, 2).reshape(2, 5, 192),
                                attention.proj.weight, attention.proj.bias)
            torch.testing.assert_close(attention(tokens), expected, atol=1e-6, rtol=1e-5)

    def test_forward_matches_explicit_fp64_equations(self):
        # Evaluate the equations using current tensors, independently of the
        # optimized modules and seed-dependent historical parameter values.
        model = DeiTTiny().eval()
        image = torch.linspace(-1., 1., 3 * 224 * 224).reshape(1, 3, 224, 224)
        with torch.no_grad():
            state = {name: value.double() for name, value in model.state_dict().items()}
            patches = F.unfold(image.double(), kernel_size=16, stride=16).transpose(1, 2)
            patches = F.linear(patches, state["patch_embed.weight"].reshape(192, -1), state["patch_embed.bias"])
            tokens = torch.cat((state["cls_token"], patches), dim=1) + state["pos_embed"]

            def normalize(values, prefix):
                centered = values - values.mean(dim=-1, keepdim=True)
                return (centered * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + 1e-6)
                        * state[prefix + ".weight"] + state[prefix + ".bias"])

            for index in range(12):
                prefix = f"blocks.{index}."
                projected = F.linear(normalize(tokens, prefix + "norm1"),
                                     state[prefix + "attn.qkv.weight"], state[prefix + "attn.qkv.bias"])
                query, key, value = [part.reshape(1, 197, 3, 64).transpose(1, 2)
                                     for part in projected.chunk(3, dim=-1)]
                attention = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(64), dim=-1) @ value
                mixed = attention.transpose(1, 2).reshape(1, 197, 192)
                tokens = tokens + F.linear(mixed, state[prefix + "attn.proj.weight"], state[prefix + "attn.proj.bias"])
                hidden = F.linear(normalize(tokens, prefix + "norm2"), state[prefix + "fc1.weight"], state[prefix + "fc1.bias"])
                hidden = .5 * hidden * (1 + torch.erf(hidden / math.sqrt(2)))
                tokens = tokens + F.linear(hidden, state[prefix + "fc2.weight"], state[prefix + "fc2.bias"])
            expected = F.linear(normalize(tokens, "norm")[:, 0], state["head.weight"], state["head.bias"])
            # Allow FP32 rounding through twelve residual blocks versus FP64.
            torch.testing.assert_close(model(image).double(), expected, atol=1e-6, rtol=1e-5)

    def test_frozen_scope_gradients_and_optimizer_updates(self):
        # Trainability is independent of the artifact. Integration below also
        # exercises the real frozen initializer with the pretrained backbone.
        model = configure_model({"experiment": "frozen", "dropout": 0.0}, initialize=False)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 1_737)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        images = torch.linspace(-1.0, 1.0, 2 * 3 * 224 * 224).reshape(2, 3, 224, 224)
        F.cross_entropy(model(images), torch.tensor([0, 8])).backward()
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("head."), name)
            self.assertEqual(parameter.grad is not None, name.startswith("head."), name)
        optimizer.step()
        for name, parameter in model.named_parameters():
            self.assertEqual(torch.equal(parameter, before[name]), not name.startswith("head."), name)


class WeightValidationChecks(unittest.TestCase):

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4531)
        self.model = DeiTTiny()

    def synthetic_source(self):
        state = self.model.state_dict()
        source = {key: torch.full_like(state[name], (index + 1) / 1000)
                  for index, (name, key) in enumerate(backbone_mapping().items())}
        source.update({"head.weight": torch.zeros(1000, 192), "head.bias": torch.zeros(1000)})
        return source

    def load_synthetic_source(self, source, directory):
        path = Path(directory) / "synthetic.pt"
        torch.save(source, path)
        load_pretrained_weights(self.model, path)

    def test_complete_mapping_and_fresh_head(self):
        source = self.synthetic_source()
        head = {name: tensor.clone() for name, tensor in self.model.head.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            self.load_synthetic_source(source, directory)
            self.load_synthetic_source({name: tensor for name, tensor in source.items()
                                        if not name.startswith("head.")}, directory)
        mapping = backbone_mapping()
        self.assertEqual(len(mapping), 150)
        self.assertEqual(set(self.model.state_dict()), set(mapping) | {"head.weight", "head.bias"})
        for name, key in mapping.items():
            self.assertTrue(torch.equal(self.model.state_dict()[name], source[key]), name)
        for name, tensor in self.model.head.state_dict().items():
            self.assertTrue(torch.equal(tensor, head[name]), name)

    def test_incompatible_and_incomplete_weights_are_rejected(self):
        source = self.synthetic_source()
        cases = {
            "missing backbone": (ValueError, "Missing pretrained tensor: patch_embed.proj.weight"),
            "extra tensor": (ValueError, "Unexpected pretrained tensors"),
            "wrong shape": (ValueError, "Incompatible pretrained tensor"),
            "wrong dtype": (ValueError, "Incompatible pretrained tensor"),
            "wrong type": (ValueError, "Incompatible pretrained tensor"),
        }
        with tempfile.TemporaryDirectory() as directory:
            for case, (error, message) in cases.items():
                malformed = dict(source)
                if case == "missing backbone":
                    del malformed["patch_embed.proj.weight"]
                elif case == "extra tensor":
                    malformed["unexpected"] = torch.zeros(1)
                elif case == "wrong shape":
                    malformed["patch_embed.proj.weight"] = torch.zeros(192, 3, 8, 8)
                elif case == "wrong dtype":
                    malformed["patch_embed.proj.weight"] = malformed["patch_embed.proj.weight"].double()
                else:
                    malformed["patch_embed.proj.weight"] = None
                with self.subTest(case=case), self.assertRaisesRegex(error, message):
                    self.load_synthetic_source(malformed, directory)


class PretrainedIntegrationChecks(unittest.TestCase):

    def test_pretrained_backbone_and_initializers(self):
        if not PRETRAINED_WEIGHTS.is_file():
            self.skipTest(f"Local ImageNet tensor checkpoint required: {PRETRAINED_WEIGHTS}")
        torch.set_num_threads(2)
        source = torch.load(PRETRAINED_WEIGHTS, weights_only=True, map_location="cpu")
        mapping = backbone_mapping()
        self.assertEqual(set(source), set(mapping.values()) | {"head.weight", "head.bias"})
        self.assertEqual(tuple(source["head.weight"].shape), (1000, 192))
        torch.manual_seed(4531)
        scratch = DeiTTiny()
        head = {name: tensor.clone() for name, tensor in scratch.head.state_dict().items()}
        for experiment in ("pretrained", "frozen"):
            with self.subTest(experiment=experiment):
                torch.manual_seed(4531)
                model = configure_model({"experiment": experiment, "dropout": 0.0,
                                         "weights": str(PRETRAINED_WEIGHTS)})
                for name, key in mapping.items():
                    value = model.state_dict()[name]
                    self.assertEqual(value.shape, source[key].shape, name)
                    self.assertEqual(value.dtype, source[key].dtype, name)
                    self.assertTrue(torch.equal(value, source[key]), name)
                for name, tensor in model.head.state_dict().items():
                    self.assertTrue(torch.equal(tensor, head[name]), name)
                trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
                expected = set(model.state_dict()) if experiment == "pretrained" else {"head.weight", "head.bias"}
                self.assertEqual(trainable, expected)
                self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),
                                 5_526_153 if experiment == "pretrained" else 1_737)
                model.eval()
                with torch.no_grad():
                    logits = model(torch.linspace(-1., 1., 3 * 224 * 224).reshape(1, 3, 224, 224))
                self.assertEqual(tuple(logits.shape), (1, 9))
                self.assertTrue(torch.isfinite(logits).all())
