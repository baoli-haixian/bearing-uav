"""Run one real-data forward/backward step for experiment F + SG-CDCA."""

import argparse

import torch
import torch.nn.functional as F

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    RSBlockDatasetPA_v3q,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_csv', required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    name = 'PARCASGM_v5a_GlobalRST_PosPrior_SGCDCA'
    dataset = RSBlockDatasetPA_v3q(args.metadata_csv, is_train=False)
    samples = [dataset[index] for index in range(args.batch_size)]
    patches = torch.stack([sample['patches'] for sample in samples])
    positions = torch.stack([sample['coords'] for sample in samples])
    headings = torch.stack([sample['agl_coords'] for sample in samples])
    device = torch.device(args.device)
    patches = patches.to(device)
    positions = positions.to(device)
    headings = headings.to(device)

    model = MODEL_CLASS_DICT[name](**MODEL_KEYWARDS_DICT[name]).to(device).train()
    position_pred, heading_pred, auxiliary = model(patches, return_aux=True)
    position_loss = F.smooth_l1_loss(position_pred, positions)
    heading_loss = model.compute_heading_losses(auxiliary, headings)['total']
    loss = 0.8 * position_loss + 0.2 * heading_loss
    loss.backward()
    direction_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.sgcdca_heading.parameters()
        if parameter.grad is not None
    )
    print(
        {
            'position_shape': tuple(position_pred.shape),
            'heading_shape': tuple(heading_pred.shape),
            'joint_logits_shape': tuple(auxiliary['joint_logits'].shape),
            'loss': float(loss.detach()),
            'sgcdca_gradient': direction_gradient,
            'finite': bool(torch.isfinite(loss)),
        }
    )


if __name__ == '__main__':
    main()
