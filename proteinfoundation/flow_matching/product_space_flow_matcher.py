from typing import Callable, Dict, Optional, Tuple, Union

import lightning as L
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from proteinfoundation.flow_matching.rdn_flow_matcher import RDNFlowMatcher

FLOW_MATCHER_FACTORY = {
    "bb_ca": RDNFlowMatcher,
    "local_latents": RDNFlowMatcher,
}  # maps data modality to correspodning class


class ProductSpaceFlowMatcher(L.LightningModule):
    """
    Base class for flow matcher. Most of the methods in this class are abstract methods
    that should be implemented by classes that inheret from this one.
    """

    def __init__(self, cfg_exp: Dict):
        super().__init__()
        self.cfg_exp = cfg_exp
        self.data_modes = [m for m in self.cfg_exp.product_flowmatcher]
        self.base_flow_matchers = self.get_base_flow_matchers()

    def get_base_flow_matchers(self):
        """Constructs all necessary flow matchers."""
        return {
            m: FLOW_MATCHER_FACTORY[m](**self.cfg_exp.product_flowmatcher[m])
            for m in self.data_modes
        }

    # ... (中间省略的方法: _apply_mask, sample_noise, interpolate, process_batch, corrupt_batch, sample_t 保持不变) ...

    def _apply_mask(
        self, x: Dict[str, Tensor], mask: Optional[Bool[Tensor, "* n"]] = None
    ):
        x = {
            data_mode: self.base_flow_matchers[data_mode]._apply_mask(
                x=x[data_mode],
                mask=mask,
            )
            for data_mode in self.data_modes
        }
        return x

    def sample_noise(
        self,
        n: int,
        shape: Tuple = tuple(),
        device: Optional[torch.device] = None,
        mask: Optional[Bool[Tensor, "* n"]] = None,
    ) -> Dict[str, Tensor]:
        x = {
            data_mode: self.base_flow_matchers[data_mode].sample_noise(
                n=n,
                shape=shape,
                device=device,
                mask=mask,
            )
            for data_mode in self.data_modes
        }
        return x

    def interpolate(
        self,
        x_0: Dict[str, torch.Tensor],
        x_1: Dict[str, torch.Tensor],
        t: Dict[str, Float[Tensor, "*"]],
        mask: Optional[Bool[Tensor, "* n"]] = None,
    ) -> Dict[str, torch.Tensor]:
        x_t = {
            data_mode: self.base_flow_matchers[data_mode].interpolate(
                x_0=x_0[data_mode],
                x_1=x_1[data_mode],
                t=t[data_mode],
                mask=mask,
            )
            for data_mode in self.data_modes
        }
        return x_t

    def process_batch(
        self, batch: Dict
    ) -> Tuple[Tensor, Tensor, Tuple, int, torch.dtype]:
        coors_tensor = batch["coords"]  # [b, n, 37, 3]
        device = coors_tensor.device
        dtype = coors_tensor.dtype
        batch_shape = coors_tensor.shape[:-3]
        n = coors_tensor.shape[-3]
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
        x_1 = self._apply_mask(x=batch["x_1"], mask=mask)
        return (x_1, mask, batch_shape, n, dtype, device)

    def corrupt_batch(
        self,
        batch: Dict,
    ) -> Dict:
        x_1, mask, batch_shape, n, dtype, device = self.process_batch(batch)
        t = self.sample_t(shape=batch_shape, device=device)
        x_0 = self.sample_noise(n=n, shape=batch_shape, mask=mask, device=device)
        x_t = self.interpolate(x_0=x_0, x_1=x_1, t=t, mask=mask)
        batch["x_0"] = x_0
        batch["x_1"] = x_1
        batch["x_t"] = x_t
        batch["t"] = t
        batch["mask"] = mask
        return batch

    def sample_t(
        self, shape: tuple, device: torch.device
    ) -> Dict[str, Float[Tensor, "*shape"]]:
        t = {
            data_mode: _sample_t(
                cfg_t_dist=self.cfg_exp.loss.t_distribution[data_mode],
                shape=shape,
                device=device,
            )
            for data_mode in self.data_modes
        }
        if self.cfg_exp.loss.t_distribution.shared_groups is None:
            return t
        for t_share_modes in self.cfg_exp.loss.t_distribution.shared_groups:
            base = t_share_modes[0]
            shared_t = t[base]
            for data_mode in t_share_modes[1:]:
                t[data_mode] = shared_t
        return t

    def compute_loss(
        self, 
        batch: Dict, 
        nn_out: Dict[str, Dict[str, Tensor]],
        loss_mask: Optional[Tensor] = None  # [MODIFIED] Add loss_mask arg
    ) -> Dict[str, Float[Tensor, "*"]]:
        """
        Computes training loss.
        """
        # Pass loss_mask to compute_fm_loss
        fm_loss = self.compute_fm_loss(batch, nn_out, loss_mask=loss_mask)
        aux_loss = self.compute_aux_loss(batch, nn_out)
        loss = {**fm_loss, **aux_loss}
        return loss

    def compute_fm_loss(
        self,
        batch: Dict,
        nn_out: Dict[str, Dict[str, Tensor]],
        loss_mask: Optional[Tensor] = None, # [MODIFIED] Add loss_mask arg
    ) -> Dict[str, Float[Tensor, "*"]]:
        """
        Computes flow matching loss.
        """
        # Use provided loss_mask if available, otherwise use batch mask (padding mask)
        mask_to_use = loss_mask if loss_mask is not None else batch["mask"]

        loss = {
            data_mode: self.base_flow_matchers[data_mode].compute_fm_loss(
                x_0=batch["x_0"][data_mode],
                x_1=batch["x_1"][data_mode],
                x_t=batch["x_t"][data_mode],
                mask=mask_to_use,  # [MODIFIED] Use the specific mask
                t=batch["t"][data_mode],
                nn_out=nn_out[data_mode],
            )
            for data_mode in self.data_modes
        }
        return loss

    def compute_aux_loss(
        self,
        batch: Dict,
        nn_out: Dict[str, Dict[str, Tensor]],
    ) -> Dict[str, Float[Tensor, "*"]]:
        # ... (Same as original) ...
        losses = {}
        if "x_motif" in batch:
            motif_mask = batch["motif_mask"]
            mask_losses = motif_mask.sum(-1).bool()
            motif_loss = {
                data_mode: self.base_flow_matchers[data_mode].compute_fm_loss(
                    x_0=batch["x_0"][data_mode],
                    x_1=batch["x_1"][data_mode],
                    x_t=batch["x_t"][data_mode],
                    mask=mask_losses,
                    t=batch["t"][data_mode],
                    nn_out=nn_out[data_mode],
                )
                for data_mode in self.data_modes
            }
            for data_mode in motif_loss:
                losses[data_mode + f"motif_loss_now_justlog"] = motif_loss[data_mode]
        return losses

    # ... (simulation_step, nn_out_to_clean..., etc. keep as is) ...
    def simulation_step(
        self,
        x_t: Dict[str, torch.Tensor],
        nn_out: Dict[str, Dict[str, torch.Tensor]],
        t: Dict[str, Float[Tensor, "*"]],
        dt: Dict[str, float],
        gt: Dict[str, float],
        simulation_step_params: Dict[str, Dict],
        mask: Optional[Bool[Tensor, "* n"]] = None,
    ) -> Dict[str, torch.Tensor]:
        x_updated = {
            data_mode: self.base_flow_matchers[data_mode].simulation_step(
                x_t=x_t[data_mode],
                nn_out=nn_out[data_mode],
                t=t[data_mode],
                dt=dt[data_mode],
                gt=gt[data_mode],
                simulation_step_params=simulation_step_params[data_mode],
                mask=mask,
            )
            for data_mode in self.data_modes
        }
        return x_updated

    def nn_out_to_clean_sample_prediction(self, batch: Dict, nn_out: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        nn_out = self.nn_out_add_clean_sample_prediction(batch=batch, nn_out=nn_out)
        return {data_mode: nn_out[data_mode]["x_1"] for data_mode in self.data_modes}

    def nn_out_add_clean_sample_prediction(self, batch: Dict, nn_out: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        for data_mode in self.data_modes:
            nn_out[data_mode] = self.base_flow_matchers[data_mode].nn_out_add_clean_sample_prediction(
                x_t=batch["x_t"][data_mode],
                t=batch["t"][data_mode],
                mask=batch["mask"],
                nn_out=nn_out[data_mode],
            )
        return nn_out

    def nn_out_add_simulation_tensor(self, batch: Dict, nn_out: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        for data_mode in self.data_modes:
            nn_out[data_mode] = self.base_flow_matchers[data_mode].nn_out_add_simulation_tensor(
                x_t=batch["x_t"][data_mode],
                t=batch["t"][data_mode],
                mask=batch["mask"],
                nn_out=nn_out[data_mode],
            )
        return nn_out

    def nn_out_add_guided_simulation_tensor(
        self,
        nn_out: Dict[str, Dict[str, torch.Tensor]],
        nn_out_ag: Union[Dict[str, Dict[str, torch.Tensor]], None],
        nn_out_ucond: Union[Dict[str, Dict[str, torch.Tensor]], None],
        guidance_w: float,
        ag_ratio: float,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        for data_mode in self.data_modes:
            nn_out[data_mode] = self.base_flow_matchers[data_mode].nn_out_add_guided_simulation_tensor(
                nn_out=nn_out[data_mode],
                nn_out_ag=nn_out_ag[data_mode] if nn_out_ag else None,
                nn_out_ucond=nn_out_ucond[data_mode] if nn_out_ucond else None,
                guidance_w=guidance_w,
                ag_ratio=ag_ratio,
            )
        return nn_out

    def get_clean_pred_n_guided_vector(
        self,
        batch: Dict,
        predict_for_sampling: Callable,
        guidance_w: float,
        ag_ratio: float,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        def _add_clean_n_sim_tensor(batch, mode):
            nn_out_dict = predict_for_sampling(batch, mode=mode)
            nn_out_dict = self.nn_out_add_clean_sample_prediction(batch, nn_out_dict)
            nn_out_dict = self.nn_out_add_simulation_tensor(batch, nn_out_dict)
            return nn_out_dict

        nn_out = _add_clean_n_sim_tensor(batch, mode="full")
        nn_out_ag = None
        nn_out_ucond = None
        if guidance_w != 1.0:
            if ag_ratio > 0.0:
                nn_out_ag = _add_clean_n_sim_tensor(batch, mode="ag")
            if ag_ratio < 1.0:
                nn_out_ucond = _add_clean_n_sim_tensor(batch, mode="ucond")

        nn_out = self.nn_out_add_guided_simulation_tensor(
            nn_out=nn_out,
            nn_out_ag=nn_out_ag,
            nn_out_ucond=nn_out_ucond,
            guidance_w=guidance_w,
            ag_ratio=ag_ratio,
        )
        return nn_out

    def full_simulation(
        self,
        batch: Dict,
        predict_for_sampling: Callable,
        nsteps: int,
        nsamples: int,
        n: int,
        self_cond: bool,
        sampling_model_args: Dict[str, Dict],
        device: torch.device,
        save_trajectory_every: int = 0,
        guidance_w: float = 1.0,
        ag_ratio: float = 0.0,
        # [MODIFIED] Added fixed_context and fixed_mask for hard inpainting support
        fixed_context: Optional[Dict[str, Tensor]] = None,
        fixed_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Generates samples by simulating the ODE. Supports Hard Inpainting/Repainting.
        """
        # overwrite n with the correct shape of coors_nm in batch
        for key, value in batch.items():
            if (
                isinstance(value, torch.Tensor)
                and value.dim() > 0
                and value.size(0) == 1
            ):
                batch[key] = value.squeeze(0)

        if "mask" in batch and batch["mask"] is not None:
            mask = batch["mask"]
        else:
            mask = torch.ones(nsamples, n).long().bool().to(device)
        assert mask.shape == (nsamples, n)
        
        if save_trajectory_every > 0:
            [{} for _ in range(nsamples)]

        ts = {
            data_mode: get_schedule(
                mode=sampling_model_args[data_mode]["schedule"]["mode"],
                nsteps=int(nsteps),
                p1=sampling_model_args[data_mode]["schedule"]["p"],
            )
            for data_mode in self.data_modes
        }

        # [MODIFIED] Only needed if you want to support SDE, but let's keep it for compatibility
        gt = {
            data_mode: get_gt(
                t=ts[data_mode][:-1],
                mode=sampling_model_args[data_mode]["gt"]["mode"],
                param=sampling_model_args[data_mode]["gt"]["p"],
                clamp_val=sampling_model_args[data_mode]["gt"]["clamp_val"],
            )
            for data_mode in self.data_modes
        }

        # [MODIFIED] Prepare noise for inpainting
        # If we have a fixed context (clean structure x_1), we need to sample noise x_0
        # that corresponds to it, so we can interpolate.
        with torch.no_grad():
            x = self.sample_noise(
                n, shape=(nsamples,), device=device, mask=mask,
            )
            
            # [MODIFIED] Initialize sampling for Inpainting
            # If fixed_context is provided, we should ideally start the chain 
            # with the correct noise for the fixed part.
            # However, standard Flow Matching usually starts from pure noise.
            # We will handle the constraint at each step (Repainting).
            
            # Capture the starting noise x_0 for repainting interpolation
            x_0_noise = {k: v.clone() for k, v in x.items()}

            for step in range(nsteps):
                t = {
                    data_mode: ts[data_mode][step] * torch.ones(nsamples, device=device)
                    for data_mode in self.data_modes
                }
                
                # [MODIFIED] We need t_next for repainting
                t_next = {
                    data_mode: ts[data_mode][step + 1] * torch.ones(nsamples, device=device)
                    for data_mode in self.data_modes
                }
                
                dt = {
                    data_mode: ts[data_mode][step + 1] - ts[data_mode][step]
                    for data_mode in self.data_modes
                }
                gt_step = {
                    data_mode: gt[data_mode][step] for data_mode in self.data_modes
                }

                batch["x_t"] = x
                batch["t"] = t
                batch["mask"] = mask
                
                if step > 0 and self_cond:
                    batch["x_sc"] = x_1_pred

                nn_out = self.get_clean_pred_n_guided_vector(
                    batch=batch,
                    predict_for_sampling=predict_for_sampling,
                    guidance_w=guidance_w,
                    ag_ratio=ag_ratio,
                )

                x_1_pred = self.nn_out_to_clean_sample_prediction(
                    batch=batch, nn_out=nn_out
                )

                simulation_step_params = {
                    data_mode: sampling_model_args[data_mode]["simulation_step_params"]
                    for data_mode in self.data_modes
                }
                
                # 1. Update x_t -> x_{t+1} using the model prediction (Euler step)
                x = self.simulation_step(
                    x_t=x,
                    nn_out=nn_out,
                    t=t,
                    dt=dt,
                    gt=gt_step,
                    mask=mask,
                    simulation_step_params=simulation_step_params,
                )

                # 2. [MODIFIED] Hard Constraint / Repainting
                # If we have a known structure (fixed_context), we force the fixed atoms
                # to lie on their correct trajectory x_t = (1-t)*x_0 + t*x_1 (Optimal Transport)
                if fixed_context is not None and fixed_mask is not None:
                    for dm in self.data_modes:
                        if dm in fixed_context:
                            # Clean target structure (x_1) for fixed regions
                            x_1_fixed = fixed_context[dm].to(device)
                            # Initial noise (x_0) for fixed regions
                            x_0_fixed = x_0_noise[dm].to(device)
                            
                            # Get the time scalar for the *next* step (where we just arrived)
                            # t_next is shape [B], we need broadcasting
                            t_val = t_next[dm].view(-1, 1, 1) if x[dm].ndim == 3 else t_next[dm].view(-1, 1, 1, 1)

                            # Compute the correct noisy state at t_{step+1}
                            # OT path: x_t = (1 - t) * x_0 + t * x_1
                            x_fixed_noisy = (1.0 - t_val) * x_0_fixed + t_val * x_1_fixed
                            
                            # Apply mask: 
                            # mask=1 (True) -> Fixed Motif -> Use x_fixed_noisy
                            # mask=0 (False) -> Generated CDR -> Use x (model output)
                            
                            # Handle mask dimensions
                            m = fixed_mask.to(device)
                            if x[dm].ndim == 3: # [B, N, C]
                                m = m.unsqueeze(-1) 
                            elif x[dm].ndim == 4: # [B, N, C, D]
                                m = m.unsqueeze(-1).unsqueeze(-1)
                            
                            # Overwrite
                            x[dm] = m * x_fixed_noisy + (~m) * x[dm]

            additional_info = {
                "mask": mask,
            }
            return x, additional_info

# ... (Helper functions get_gt, get_schedule, _sample_t keep as is) ...
def get_gt(t, mode, param, clamp_val=None, eps=1e-2):
    # ... same as original ...
    def transform_gt(gt, f_pow=1.0):
        if f_pow == 1.0: return gt
        log_gt = torch.log(gt)
        mean_log_gt = torch.mean(log_gt)
        log_gt_centered = log_gt - mean_log_gt
        normalized = torch.nn.functional.sigmoid(log_gt_centered)
        normalized = normalized**f_pow
        log_gt_centered_rec = torch.logit(normalized, eps=1e-6)
        log_gt_rec = log_gt_centered_rec + mean_log_gt
        gt_rec = torch.exp(log_gt_rec)
        return gt_rec
    t = torch.clamp(t, 0, 1 - 1e-5)
    if mode == "1-t/t":
        num = 1.0 - t; den = t; gt = num / (den + eps)
    elif mode == "tan":
        num = torch.sin((1.0 - t) * torch.pi / 2.0)
        den = torch.cos((1.0 - t) * torch.pi / 2.0)
        gt = (torch.pi / 2.0) * num / (den + eps)
    elif mode == "1/t":
        num = 1.0; den = t; gt = num / (den + eps)
    else: raise NotImplementedError(f"gt not implemented {mode}")
    gt = transform_gt(gt, f_pow=param)
    gt = torch.clamp(gt, 0, clamp_val)
    return gt

def get_schedule(mode, nsteps, *, p1=None, eps=1e-5):
    # ... same as original ...
    if mode == "uniform":
        t = torch.linspace(0, 1, nsteps + 1)
        return t
    elif mode == "power":
        assert p1 is not None
        t = torch.linspace(0, 1, nsteps + 1)
        t = t**p1
        return t
    elif mode == "log":
        assert p1 is not None; assert p1 > 0
        t = 1.0 - torch.logspace(-p1, 0, nsteps + 1).flip(0)
        t = t - torch.min(t); t = t / torch.max(t)
        return t
    else: raise IOError(f"Schedule mode not recognized {mode}")

def _sample_t(cfg_t_dist, shape, device=torch.device):
    # ... same as original ...
    if cfg_t_dist.name == "uniform":
        t_max = cfg_t_dist.p2
        return torch.rand(shape, device=device) * t_max
    elif cfg_t_dist.name == "logit-normal":
        mean = cfg_t_dist.p1; std = cfg_t_dist.p2
        noise = torch.randn(shape, device=device) * std + mean
        return torch.nn.functional.sigmoid(noise)
    elif cfg_t_dist.name == "beta":
        p1 = cfg_t_dist.p1; p2 = cfg_t_dist.p2
        dist = torch.distributions.beta.Beta(p1, p2)
        return dist.sample(shape).to(device)
    elif cfg_t_dist.name == "mix_unif_beta":
        p1 = cfg_t_dist.p1; p2 = cfg_t_dist.p2; p3 = cfg_t_dist.p3
        dist = torch.distributions.beta.Beta(p1, p2)
        samples_beta = dist.sample(shape).to(device)
        samples_uniform = torch.rand(shape, device=device)
        u = torch.rand(shape, device=device)
        return torch.where(u < p3, samples_uniform, samples_beta)
    else: raise NotImplementedError