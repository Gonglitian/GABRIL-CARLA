#!/usr/bin/env python3
"""
Visualization utilities for training data in BC trainer
Generates GIFs showing:
1. Input images 
2. Gaze heatmaps
3. Overlay of images and heatmaps

Basic Usage:
------------
# 1. Visualize real gaze data from HDF5 (default - sequential demos)
python vlm_gaze/data_utils/train_data_viz.py --num_demos 1 --frames_per_demo 300

# 2. Randomly select demos for visualization
python vlm_gaze/data_utils/train_data_viz.py --random_demos --num_demos 3 --frames_per_demo 20

# 3. Random selection with fixed seed (reproducible)
python vlm_gaze/data_utils/train_data_viz.py --random_demos --seed 42 --num_demos 5

# 4. Visualize pseudo gaze data
python vlm_gaze/data_utils/train_data_viz.py --gaze_key gaze_coords_gaze_pseudo --output_dir outputs/viz_pseudo

# 5. Visualize with different gaze sources
python vlm_gaze/data_utils/train_data_viz.py --gaze_key gaze_coords_filter_dynamic --num_demos 5

# 6. Test with synthetic data (no HDF5 required)
python vlm_gaze/data_utils/train_data_viz.py --test_synthetic

# 7. Custom HDF5 file path
python vlm_gaze/data_utils/train_data_viz.py --data_path /path/to/your/data.hdf5 --num_demos 2

Command Line Arguments:
----------------------
--data_path: Path to HDF5 dataset file (default: /data3/vla-reasoning/dataset/bench2drive220_robomimic.hdf5)
--output_dir: Output directory for GIFs (default: outputs/training_viz_real)
--num_demos: Number of demos to visualize (default: 3)
--frames_per_demo: Number of frames per demo (default: 20)
--gaze_key: Gaze coordinate key to use (default: gaze_coords)
            Options: gaze_coords, gaze_coords_gaze_pseudo, gaze_coords_filter_dynamic, 
                    gaze_coords_non_filter, gaze_coords_gaze
--use_pseudo_gaze: Use pseudo gaze (overrides gaze_key to gaze_coords_gaze_pseudo)
--random_demos: Randomly select demos instead of sequential selection
--seed: Random seed for reproducible demo selection (e.g., 42)
--test_synthetic: Test with synthetic data only

Output:
-------
Generates GIF files in the output directory with three panels:
- Left: Original input image
- Middle: Gaze heatmap visualization  
- Right: Overlay of image and heatmap

Note: Requires vlm-gabril conda environment to be activated
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io


class TrainingDataVisualizer:
    """
    Visualizes training data from BC trainer including images, heatmaps and overlays
    """
    
    def __init__(
        self,
        output_dir: str = "outputs/training_viz",
        max_frames: int = 999,
        fps: int = 20,
        overlay_alpha: float = 0.4
    ):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Directory to save output GIFs
            max_frames: Maximum number of frames to include in GIF
            fps: Frames per second for GIF
            overlay_alpha: Alpha value for heatmap overlay (0=transparent, 1=opaque)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames
        self.fps = fps
        self.overlay_alpha = overlay_alpha
        self.frame_duration = 1000 // fps  # Duration per frame in milliseconds
        
    def visualize_batch(
        self,
        obs_images: torch.Tensor,
        gaze_heatmaps: torch.Tensor,
        batch_idx: int = 0,
        epoch: int = 0,
        save_individual: bool = True
    ) -> Optional[Path]:
        """
        Visualize a batch of training data
        
        Args:
            obs_images: Input images tensor [B, C, H, W] or [B, S, C, H, W]
            gaze_heatmaps: Gaze heatmap tensor [B, C, H, W] or [B, S, 1, H, W]
            batch_idx: Index of current batch
            epoch: Current training epoch
            save_individual: Whether to save individual frames as images
            
        Returns:
            Path to saved GIF file if successful
        """
        # Move tensors to CPU and convert to numpy
        obs_images = obs_images.detach().cpu()
        gaze_heatmaps = gaze_heatmaps.detach().cpu()
        
        # Handle frame stacking - extract individual frames from all samples in batch
        frames_img = []
        frames_hm = []
        
        # Process all samples in the batch
        batch_size = obs_images.shape[0]
        for sample_idx in range(min(batch_size, self.max_frames)):
            obs_img = obs_images[sample_idx]  # [C, H, W] or [S, C, H, W]
            gaze_hm = gaze_heatmaps[sample_idx]  # [C, H, W] or [S, 1, H, W]
            
            # Check if we have stacked frames
            if obs_img.dim() == 3:
                # Single frame or stacked frames in channel dimension
                C = obs_img.shape[0]
                
                # Determine number of stacked frames
                if C % 3 == 0:
                    # RGB stacked frames
                    num_stack = C // 3
                    for i in range(num_stack):
                        frame = obs_img[i*3:(i+1)*3]  # Extract RGB channels
                        frames_img.append(frame)
                elif C == 1 or C == 2:
                    # Grayscale (possibly stacked)
                    for i in range(C):
                        frame = obs_img[i:i+1].repeat(3, 1, 1)  # Convert to RGB for visualization
                        frames_img.append(frame)
                else:
                    # Unknown format, take as is
                    frames_img.append(obs_img[:3] if C >= 3 else obs_img.repeat(3, 1, 1))
                    
                # Process heatmaps
                hm_C = gaze_hm.shape[0]
                for i in range(min(hm_C, num_stack if C % 3 == 0 else 1)):
                    frames_hm.append(gaze_hm[i:i+1])
                    
            else:
                # Explicit time/stack dimension [S, C, H, W]
                S = obs_img.shape[0]
                for i in range(S):
                    frame = obs_img[i]
                    if frame.shape[0] == 1:
                        frame = frame.repeat(3, 1, 1)  # Grayscale to RGB
                    elif frame.shape[0] > 3:
                        frame = frame[:3]  # Take first 3 channels
                    frames_img.append(frame)
                    
                    if i < gaze_hm.shape[0]:
                        frames_hm.append(gaze_hm[i, 0:1] if gaze_hm.dim() == 4 else gaze_hm[i:i+1])
        
        # Ensure we have matching number of frames
        min_frames = min(len(frames_img), len(frames_hm), self.max_frames)
        frames_img = frames_img[:min_frames]
        frames_hm = frames_hm[:min_frames]
        
        if len(frames_img) == 0:
            print(f"Warning: No frames to visualize for batch {batch_idx}")
            return None
            
        # Create visualizations
        pil_images = []
        individual_dir = self.output_dir / f"epoch_{epoch:04d}" / f"batch_{batch_idx:06d}" if save_individual else None
        if individual_dir:
            individual_dir.mkdir(parents=True, exist_ok=True)
        
        for frame_idx, (img_frame, hm_frame) in enumerate(zip(frames_img, frames_hm)):
            # Create composite visualization
            fig = self._create_composite_figure(img_frame, hm_frame, frame_idx)
            
            # Convert matplotlib figure to PIL Image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)
            pil_img = Image.open(buf).copy()
            pil_images.append(pil_img)
            buf.close()
            plt.close(fig)
            
            # Save individual frame if requested
            if individual_dir:
                pil_img.save(individual_dir / f"frame_{frame_idx:04d}.png")
        
        # Save as GIF
        gif_path = self.output_dir / f"epoch_{epoch:04d}_batch_{batch_idx:06d}.gif"
        if len(pil_images) > 1:
            pil_images[0].save(
                gif_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=self.frame_duration,
                loop=0
            )
        else:
            # Single frame - save as static image
            pil_images[0].save(gif_path)
            
        print(f"Saved visualization to {gif_path} ({len(pil_images)} frames)")
        return gif_path
    
    def _create_composite_figure(
        self,
        img_tensor: torch.Tensor,
        hm_tensor: torch.Tensor,
        frame_idx: int
    ) -> plt.Figure:
        """
        Create a composite figure with image, heatmap, and overlay
        
        Args:
            img_tensor: Image tensor [C, H, W] 
            hm_tensor: Heatmap tensor [1, H, W]
            frame_idx: Current frame index
            
        Returns:
            Matplotlib figure with 3 subplots
        """
        # Convert tensors to numpy
        img_np = img_tensor.numpy()
        hm_np = hm_tensor.numpy()
        
        # Normalize image to [0, 1] if needed
        if img_np.min() < 0:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        elif img_np.max() > 1:
            img_np = img_np / 255.0
            
        # Ensure image is in HWC format for display
        if img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
        
        # Squeeze heatmap to 2D
        if hm_np.shape[0] == 1:
            hm_np = hm_np[0]
            
        # Normalize heatmap to [0, 1]
        if hm_np.max() > hm_np.min():
            hm_np = (hm_np - hm_np.min()) / (hm_np.max() - hm_np.min())
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Original Image
        axes[0].imshow(img_np if img_np.shape[-1] == 3 else img_np[:, :, 0], cmap='gray' if img_np.shape[-1] == 1 else None)
        axes[0].set_title(f"Input Image (Frame {frame_idx})")
        axes[0].axis('off')
        
        # 2. Gaze Heatmap
        im = axes[1].imshow(hm_np, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f"Gaze Heatmap")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. Overlay
        axes[2].imshow(img_np if img_np.shape[-1] == 3 else img_np[:, :, 0], cmap='gray' if img_np.shape[-1] == 1 else None)
        # Apply colormap to heatmap for overlay
        hm_colored = cm.hot(hm_np)
        # Create alpha mask based on heatmap intensity
        alpha_mask = hm_np * self.overlay_alpha
        hm_colored[:, :, 3] = alpha_mask
        axes[2].imshow(hm_colored)
        axes[2].set_title(f"Overlay (alpha={self.overlay_alpha})")
        axes[2].axis('off')
        
        plt.suptitle(f"Training Data Visualization", fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def create_summary_gif(
        self,
        all_images: List[torch.Tensor],
        all_heatmaps: List[torch.Tensor],
        output_name: str = "training_summary.gif"
    ) -> Path:
        """
        Create a summary GIF from multiple batches
        
        Args:
            all_images: List of image tensors from different batches
            all_heatmaps: List of heatmap tensors from different batches
            output_name: Name of output GIF file
            
        Returns:
            Path to saved GIF file
        """
        pil_images = []
        
        for batch_idx, (imgs, hms) in enumerate(zip(all_images, all_heatmaps)):
            # Take first sample from each batch
            img = imgs[0] if imgs.dim() > 3 else imgs
            hm = hms[0] if hms.dim() > 3 else hms
            
            # Extract first frame if stacked
            if img.shape[0] > 3:
                img = img[:3]
            elif img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
                
            if hm.shape[0] > 1:
                hm = hm[0:1]
                
            # Create visualization
            fig = self._create_composite_figure(img.cpu(), hm.cpu(), batch_idx)
            
            # Convert to PIL
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)
            pil_img = Image.open(buf).copy()
            pil_images.append(pil_img)
            buf.close()
            plt.close(fig)
            
            if len(pil_images) >= self.max_frames:
                break
        
        # Save GIF
        gif_path = self.output_dir / output_name
        if len(pil_images) > 1:
            pil_images[0].save(
                gif_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=self.frame_duration,
                loop=0
            )
        else:
            pil_images[0].save(gif_path)
            
        print(f"Saved summary GIF to {gif_path} ({len(pil_images)} frames)")
        return gif_path


# Hook function to be called from train_bc.py
def visualize_training_batch(
    obs_images: torch.Tensor,
    gaze_heatmaps: torch.Tensor,
    epoch: int = 0,
    batch_idx: int = 0,
    output_dir: str = "outputs/training_viz",
    max_viz_per_epoch: int = 5
) -> Optional[Path]:
    """
    Simple hook function to visualize training data
    Can be called from train_bc.py's compute_loss method
    
    Args:
        obs_images: Input images [B, C, H, W]
        gaze_heatmaps: Gaze heatmaps [B, C, H, W]
        epoch: Current epoch
        batch_idx: Current batch index
        output_dir: Output directory
        max_viz_per_epoch: Maximum visualizations per epoch
        
    Returns:
        Path to saved visualization or None
    """
    # Only visualize first few batches of each epoch
    if batch_idx >= max_viz_per_epoch:
        return None
        
    visualizer = TrainingDataVisualizer(
        output_dir=output_dir,
        max_frames=999,
        fps=20,
        overlay_alpha=0.5
    )
    
    return visualizer.visualize_batch(
        obs_images,
        gaze_heatmaps,
        batch_idx=batch_idx,
        epoch=epoch,
        save_individual=False
    )


def visualize_from_training_data(
    data_path: str = "/data3/vla-reasoning/dataset/bench2drive220_robomimic.hdf5",
    output_dir: str = "outputs/training_viz_real",
    num_demos: int = 3,
    frames_per_demo: int = 20,
    use_pseudo_gaze: bool = False,
    gaze_key: str = "gaze_coords",
    random_demos: bool = False,
    seed: int = None
):
    """
    Load actual training data and visualize it
    
    Args:
        data_path: Path to the HDF5 dataset file
        output_dir: Output directory for visualizations
        num_demos: Number of demos to visualize
        frames_per_demo: Number of frames to visualize per demo
        use_pseudo_gaze: Whether to use pseudo gaze data
        gaze_key: Which gaze key to use (gaze_coords, gaze_coords_gaze_pseudo, etc.)
        random_demos: If True, randomly select demos instead of sequential
        seed: Random seed for reproducible demo selection
    """
    import sys
    sys.path.append('/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA')
    
    from vlm_gaze.data_utils import GazePreprocessor
    from vlm_gaze.train.common.preprocess import format_obs_image
    from einops import repeat, rearrange
    
    print(f"Loading data from {data_path}...")
    
    # Initialize preprocessor
    gaze_preprocessor = GazePreprocessor(
        img_height=180,
        img_width=320,
        gaze_sigma=15.0,
        gaze_coeff=0.8,
        maxpoints=5,
        device='cpu'
    )
    
    # Initialize visualizer
    visualizer = TrainingDataVisualizer(
        output_dir=output_dir,
        max_frames=frames_per_demo,
        fps=5,
        overlay_alpha=0.5
    )
    
    # Load HDF5 file
    import h5py
    data_file = Path(data_path)
    
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        print("Falling back to synthetic data test...")
        return test_with_synthetic_data()
    
    with h5py.File(data_file, 'r') as f:
        print(f"Dataset keys: {list(f.keys())}")
        
        # Get list of demos and filter valid ones
        if 'data' in f:
            all_demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])
            
            # Check which demos actually have data
            demo_keys = []
            for demo_key in all_demo_keys:
                try:
                    # Try to access the image data to verify it exists
                    _ = f[f'data/{demo_key}/obs/image'].shape
                    demo_keys.append(demo_key)
                except KeyError:
                    print(f"Warning: {demo_key} found in keys but has no data, skipping...")
            
            print(f"Found {len(demo_keys)} valid demos (out of {len(all_demo_keys)} total)")
            
            # Select demos - either random or sequential
            if random_demos:
                import random
                if seed is not None:
                    random.seed(seed)
                    print(f"Using random seed: {seed}")
                
                # Randomly select demos
                selected_indices = random.sample(range(len(demo_keys)), min(num_demos, len(demo_keys)))
                selected_demos = [demo_keys[i] for i in selected_indices]
                print(f"Randomly selected demos: {', '.join(selected_demos)}")
            else:
                # Sequential selection (original behavior)
                selected_demos = demo_keys[:min(num_demos, len(demo_keys))]
                print(f"Sequential demos: {', '.join(selected_demos)}")
            
            # Process selected demos
            for demo_idx, demo_key in enumerate(selected_demos):
                print(f"\nProcessing {demo_key} ({demo_idx+1}/{len(selected_demos)})...")
                
                # Load image data for this demo
                image_data = f[f'data/{demo_key}/obs/image'][:]  # [T, H, W, C]
                
                # Determine which gaze key to use
                if use_pseudo_gaze:
                    gaze_key = "gaze_coords_gaze_pseudo"
                
                # Load gaze coordinates
                gaze_path = f'data/{demo_key}/obs/{gaze_key}'
                if gaze_path in f:
                    gaze_data = f[gaze_path][:]  # [T, 10] for 5 points with x,y coords
                    print(f"Using gaze key: {gaze_key}, shape: {gaze_data.shape}")
                else:
                    print(f"Warning: {gaze_path} not found, using zeros")
                    gaze_data = np.zeros((len(image_data), 10), dtype=np.float32)
                
                # Process frames
                total_frames = len(image_data)
                # For frame stacking, we need pairs of frames
                max_frame_pairs = (total_frames - 1) // 2
                frames_to_viz = min(frames_per_demo, max_frame_pairs)
                
                # Create batch of frames for visualization
                batch_images = []
                batch_gazes = []
                
                # Collect frame pairs for visualization
                for pair_idx in range(frames_to_viz):
                    frame_idx = pair_idx * 2
                    if frame_idx + 1 >= total_frames:
                        break
                        
                    # Stack 2 consecutive frames
                    stacked_image = np.stack([
                        image_data[frame_idx],
                        image_data[frame_idx + 1]
                    ], axis=0)  # [2, H, W, C]
                    
                    # Get gaze for current frame (we use the second frame's gaze)
                    current_gaze = gaze_data[frame_idx + 1]  # [10]
                    
                    batch_images.append(stacked_image)
                    batch_gazes.append(current_gaze)
                
                if len(batch_images) == 0:
                    print(f"No frames to process for {demo_key}")
                    continue
                
                # Convert to tensors
                batch_images = torch.from_numpy(np.array(batch_images))  # [B, 2, H, W, C]
                batch_gazes = torch.from_numpy(np.array(batch_gazes))  # [B, 10]
                
                # Format images for model input
                batch_images = format_obs_image(batch_images, frame_stack=2, grayscale=False)
                
                # Reshape gaze coords from [B, 10] to [B, 5, 2]
                batch_gazes = rearrange(batch_gazes, 'b (p c) -> b p c', p=5, c=2)
                
                # Generate gaze heatmaps
                with torch.no_grad():
                    # Add time dimension for preprocessor
                    batch_gazes_for_proc = batch_gazes.unsqueeze(1)  # [B, 1, 5, 2]
                    gaze_heatmaps = gaze_preprocessor(batch_gazes_for_proc)
                    
                    if gaze_heatmaps.dim() == 5:
                        gaze_heatmaps = gaze_heatmaps[:, 0]  # Remove time dim
                    
                    # Repeat for frame stacking
                    if batch_images.shape[1] > gaze_heatmaps.shape[1]:
                        gaze_heatmaps = repeat(gaze_heatmaps, 'b 1 h w -> b s h w', s=batch_images.shape[1])
                
                # Visualize this demo
                gif_path = visualizer.visualize_batch(
                    batch_images,
                    gaze_heatmaps,
                    batch_idx=demo_idx,
                    epoch=0,
                    save_individual=False
                )
                
                if gif_path:
                    print(f"✓ Saved visualization to: {gif_path}")
        else:
            print("Error: 'data' group not found in HDF5 file")
            return False
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    return True


def test_with_synthetic_data():
    """Test with synthetic data when real data is not available"""
    print("\nTesting TrainingDataVisualizer with synthetic data...")
    
    # Create synthetic data
    batch_size = 2
    frame_stack = 2
    height, width = 180, 320
    
    # Stacked RGB images [B, stack*3, H, W]
    obs_images = torch.rand(batch_size, frame_stack * 3, height, width)
    
    # Stacked heatmaps [B, stack, H, W]
    gaze_heatmaps = torch.zeros(batch_size, frame_stack, height, width)
    
    # Add some Gaussian-like patterns
    for b in range(batch_size):
        for s in range(frame_stack):
            cx, cy = np.random.randint(50, width-50), np.random.randint(50, height-50)
            y, x = np.ogrid[:height, :width]
            mask = ((x - cx)**2 + (y - cy)**2) < 30**2
            gaze_heatmaps[b, s, mask] = 1.0
    
    # Apply Gaussian blur to heatmaps
    from scipy.ndimage import gaussian_filter
    for b in range(batch_size):
        for s in range(frame_stack):
            gaze_heatmaps[b, s] = torch.from_numpy(
                gaussian_filter(gaze_heatmaps[b, s].numpy(), sigma=10)
            )
    
    # Test visualizer
    visualizer = TrainingDataVisualizer(
        output_dir="outputs/test_training_viz",
        max_frames=20,
        fps=5,
        overlay_alpha=0.4
    )
    
    gif_path = visualizer.visualize_batch(
        obs_images,
        gaze_heatmaps,
        batch_idx=0,
        epoch=0,
        save_individual=True
    )
    
    if gif_path and gif_path.exists():
        print(f"✓ Test successful! GIF saved to: {gif_path}")
        return True
    else:
        print("✗ Test failed!")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize training data with gaze heatmaps")
    parser.add_argument('--data_path', type=str, default="/data3/vla-reasoning/dataset/bench2drive220_robomimic.hdf5",
                        help="Path to the HDF5 dataset file")
    parser.add_argument('--output_dir', type=str, default="outputs/training_viz_real",
                        help="Output directory for visualizations")
    parser.add_argument('--num_demos', type=int, default=3,
                        help="Number of demos to visualize")
    parser.add_argument('--frames_per_demo', type=int, default=20,
                        help="Number of frames to visualize per demo")
    parser.add_argument('--use_pseudo_gaze', action='store_true',
                        help="Use pseudo gaze coordinates instead of real gaze")
    parser.add_argument('--gaze_key', type=str, default="gaze_coords",
                        help="Which gaze key to use (gaze_coords, gaze_coords_gaze_pseudo, etc.)")
    parser.add_argument('--random_demos', action='store_true',
                        help="Randomly select demos instead of sequential selection")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducible demo selection")
    parser.add_argument('--test_synthetic', action='store_true',
                        help="Test with synthetic data only")
    
    args = parser.parse_args()
    
    if args.test_synthetic:
        test_with_synthetic_data()
    else:
        visualize_from_training_data(
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_demos=args.num_demos,
            frames_per_demo=args.frames_per_demo,
            use_pseudo_gaze=args.use_pseudo_gaze,
            gaze_key=args.gaze_key,
            random_demos=args.random_demos,
            seed=args.seed
        )