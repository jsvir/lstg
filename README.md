# Local Stochastic Gates Model Example (Supervised)

- Single-file python script that defines and trains Local STG NNet with Linear classifier
- Here we use [MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) subset (first 10K samples for train set and first 1K samples for test)
- The hyperparameters were not tuned to the best, you can try to modify `LocalSTG` hidden dimension (the dafault is 128), regularization loss term multiplier (the dafault is 0.5) or `Trainer` arguments.


Samples (Test ACC 86.5% on 1K subset):

| Original Image                                                                            | Sparse Image |
|-------------------------------------------------------------------------------------------|--------------|
| <img height="64" src="samples\orig_sample_3_y_0_y_hat_0_batch_idx_0.png" width="64"/>     |   <img height="64" src="samples\result_sample_3_y_0_y_hat_0_batch_idx_0.png" width="64"/> |
|<img height="64" src="samples\orig_sample_2_y_1_y_hat_1_batch_idx_0.png" width="64"/>      | <img height="64" src="samples\result_sample_2_y_1_y_hat_1_batch_idx_0.png" width="64"/>   |
| <img height="64" src="samples\orig_sample_1_y_2_y_hat_6_batch_idx_0.png" width="64"/>     |   <img height="64" src="samples\result_sample_1_y_2_y_hat_6_batch_idx_0.png" width="64"/>          |
| <img height="64" src="samples\orig_sample_18_y_3_y_hat_6_batch_idx_0.png" width="64"/>    |   <img height="64" src="samples\result_sample_18_y_3_y_hat_6_batch_idx_0.png" width="64"/>         |
| <img height="64" src="samples\orig_sample_6_y_4_y_hat_4_batch_idx_0.png" width="64"/>     |   <img height="64" src="samples\result_sample_6_y_4_y_hat_4_batch_idx_0.png" width="64"/>          |
| <img height="64" src="samples\orig_sample_8_y_5_y_hat_6_batch_idx_0.png" width="64"/>     |   <img height="64" src="samples\result_sample_8_y_5_y_hat_6_batch_idx_0.png" width="64"/>          |
| <img height="64" src="samples\orig_sample_11_y_6_y_hat_6_batch_idx_0.png" width="64"/>    |   <img height="64" src="samples\result_sample_11_y_6_y_hat_6_batch_idx_0.png" width="64"/>          |
| <img height="64" src="samples\orig_sample_17_y_7_y_hat_7_batch_idx_0.png" width="64"/>    |   <img height="64" src="samples\result_sample_17_y_7_y_hat_7_batch_idx_0.png" width="64"/>          |
| <img height="64" src="samples\orig_sample_12_y_9_y_hat_9_batch_idx_0.png" width="64"/>  |   <img height="64" src="samples\result_sample_12_y_9_y_hat_9_batch_idx_0.png" width="64"/>          |



