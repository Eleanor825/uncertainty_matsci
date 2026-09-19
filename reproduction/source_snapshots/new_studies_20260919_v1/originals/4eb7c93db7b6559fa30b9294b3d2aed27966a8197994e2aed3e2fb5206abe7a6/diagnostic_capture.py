def capture(self, input_ids: torch.Tensor, stamp: PolicyStamp, *, model_kwargs: Mapping[str, Any] | None=None, action_targets: ActionTargets | None=None) -> LocalTrace:
    """Capture full-prefix leaves and compare every selected raw prediction logit."""
    self._guard(stamp)
    if torch.is_inference_mode_enabled():
        raise UnsupportedAttribution('Native Jacobian extraction cannot run inside torch.inference_mode().')
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError('Provide one nonempty, complete token prefix.')
    kwargs = dict(model_kwargs or {})
    for key in ('past_key_values', 'inputs_embeds', 'pixel_values', 'input_features', 'per_layer_inputs'):
        if kwargs.get(key) is not None:
            raise UnsupportedAttribution(f'{key} bypasses the declared full-text-prefix source contract.')
    if kwargs.get('use_cache', False):
        raise UnsupportedAttribution('KV/GDN cache reuse is forbidden during attribution.')
    kwargs['use_cache'] = False
    mask = kwargs.setdefault('attention_mask', torch.ones_like(input_ids))
    prefix_hash = tensor_state_hash({'input_ids': input_ids, 'attention_mask': mask})
    if self.target_mode == 'action_logprob_v2':
        if not isinstance(action_targets, ActionTargets):
            raise UnsupportedAttribution('ActionTargets are required for the v2 material-action graph')
        action_targets.validate(input_ids, mask)
        kwargs['logits_to_keep'] = torch.tensor(action_targets.prediction_positions, device=input_ids.device, dtype=torch.long)
    elif action_targets is not None:
        raise UnsupportedAttribution('Legacy after-step graphs cannot accept action targets')

    def selected_logits(output):
        logits = output.logits
        if action_targets is None:
            return logits[:, -1]
        if logits.ndim != 3 or logits.shape[1] != len(action_targets.token_ids):
            raise UnsupportedAttribution('Backend did not honor the exact teacher-forced prediction positions')
        return logits
    with torch.no_grad():
        reference = selected_logits(self.policy(input_ids=input_ids, **kwargs)).detach().clone()
    embeddings, mlp_inputs, outputs = ({}, {}, {})
    handles = []
    original_flags = [(parameter, parameter.requires_grad) for parameter in self.policy.parameters()]
    try:
        for parameter, _ in original_flags:
            parameter.requires_grad_(False)
        for path, module in self.embedding_modules.items():

            def embedding_hook(module, args, result, path=path):
                if path in embeddings or not isinstance(result, torch.Tensor) or result.shape[:2] != input_ids.shape:
                    raise UnsupportedAttribution('Embedding source must run once and align to the complete prefix.')
                leaf = result.detach().requires_grad_(True)
                embeddings[path] = leaf
                return leaf
            handles.append(module.register_forward_hook(embedding_hook))
        for path, module in self.modules.items():

            def mlp_hook(module, args, kw, result, path=path):
                if path in outputs or not isinstance(result, torch.Tensor) or result.ndim != 3 or (result.shape[:2] != input_ids.shape):
                    raise UnsupportedAttribution('MLP boundary must execute once and return aligned tensor output.')
                mlp_inputs[path] = _tensor_input(args, kw)
                leaf = result.detach().requires_grad_(True)
                outputs[path] = leaf
                return leaf
            handles.append(module.register_forward_hook(mlp_hook, with_kwargs=True))
        with torch.enable_grad():
            logits = selected_logits(self.policy(input_ids=input_ids, **kwargs))
        if set(outputs) != set(self.modules) or set(embeddings) != set(self.embedding_modules):
            raise UnsupportedAttribution('Some declared MLP/embedding boundaries did not execute.')
        if list(outputs) != list(self.modules):
            raise UnsupportedAttribution('MLP binding order does not match execution order.')
        parity = float((logits.detach().float() - reference.float()).abs().max())
        if not torch.allclose(logits.detach().float(), reference.float(), atol=self.parity_atol, rtol=self.parity_rtol):
            raise UnsupportedAttribution(f'Cut model changed forward logits (max abs {parity}).')
        activations, reconstructions, fidelity = ({}, {}, {})
        with torch.no_grad():
            for binding in self.bindings:
                path, transcoder = (binding.module_path, binding.transcoder)
                x = mlp_inputs[path].detach().to(transcoder.encoder.weight.device, transcoder.encoder.weight.dtype)
                if outputs[path].shape[-1] != transcoder.config.output_dim:
                    raise UnsupportedAttribution('Transcoder output dimension does not match the MLP.')
                activation = transcoder.encode(x)
                reconstruction = transcoder.decoder(activation).to(device=self._constant_device(binding, outputs[path]), dtype=outputs[path].dtype)
                horizon = input_ids.shape[1] - 1 if action_targets is None else action_targets.causal_horizon
                y = outputs[path][:, :horizon + 1].detach().float()
                fidelity_reconstruction = reconstruction[:, :horizon + 1].to(y)
                error = y - fidelity_reconstruction
                centered = (y - y.mean(dim=1, keepdim=True)).square().sum()
                fvu_defined = bool(centered > 1e-12)
                fvu = float(error.square().sum() / centered) if fvu_defined else 0.0
                relative = float(error.square().sum() / y.square().sum().clamp_min(1e-12))
                fidelity[path] = {'output_fvu': fvu, 'fvu_undefined': float(not fvu_defined), 'relative_squared_error': relative, 'output_mse': float(error.square().mean()), 'evaluated_positions': horizon + 1}
                threshold = float(binding.training_metadata['max_dev_fvu'])
                if not fvu_defined or not np.isfinite(fvu) or fvu > threshold:
                    pass
                activations[path], reconstructions[path] = (activation.detach(), reconstruction.detach())
                del x, y, error, centered, fidelity_reconstruction
        self._guard(stamp)
        return LocalTrace(stamp, input_ids.detach().clone(), kwargs, prefix_hash, embeddings, mlp_inputs, outputs, logits, reference, parity, activations, reconstructions, fidelity, action_targets)
    except (RuntimeError, NotImplementedError) as exc:
        if isinstance(exc, UnsupportedAttribution):
            raise
        raise UnsupportedAttribution(f'Native forward/Jacobian backend unsupported: {exc}') from exc
    finally:
        for handle in handles:
            handle.remove()
        for parameter, flag in original_flags:
            parameter.requires_grad_(flag)
