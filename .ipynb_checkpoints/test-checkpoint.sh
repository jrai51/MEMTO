#!/bin/bash

# SMD
# python3 main.py --anormly_ratio 0.5 --num_epochs 100 --batch_size 256 --mode train --dataset SMD --data_path ./data/SMD/SMD/ --input_c 38 --output_c 38 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None

# python3 main.py --anormly_ratio 0.5 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset SMD  --data_path ./data/SMD/SMD/  --input_c 38 --output_c 38 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python3 main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 256  --mode test --dataset SMD  --data_path ./data/SMD/SMD/  --input_c 38 --output_c 38 --n_memory 10 --memory_initial False --phase_type test

# export PATH=$PATH:/usr/local/cuda/bin
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


python3 cuda_test.py

# Updated input_c output_c from 38 to match actual channel size of 26

# python3 main.py --anormly_ratio 0.5 --num_epochs 100 --batch_size 256 --mode train --dataset SMD --data_path ./data/SMD/SMD/ --input_c 26 --output_c 26 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None

# python3 main.py --anormly_ratio 0.5 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset SMD  --data_path ./data/SMD/SMD/  --input_c 26 --output_c 26 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python3 main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 256  --mode test --dataset SMD  --data_path ./data/SMD/SMD/  --input_c 26 --output_c 26 --n_memory 10 --memory_initial False --phase_type test




# SMAP
# python3 main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 256 --mode train --dataset SMAP --data_path ./data/SMAP/SMAP/ --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python3 main.py --anormly_ratio 1.0 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset SMAP  --data_path ./data/SMAP/SMAP/  --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python3 main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 256  --mode test --dataset SMAP  --data_path ./data/SMAP/SMAP/  --input_c 25 --output_c 25 --n_memory 10 --memory_initial False --phase_type test

# # # PSM
# python3 main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 256 --mode train --dataset PSM --data_path ./data/PSM/PSM/ --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anormly_ratio 1.0 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset PSM  --data_path ./data/PSM/PSM/  --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 256  --mode test --dataset PSM  --data_path ./data/PSM/PSM/  --input_c 25 --output_c 25 --n_memory 10 --memory_initial False --phase_type test

# PSM lower batch size
python3 main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 32 --mode train --dataset PSM --data_path ./data/PSM/PSM/ --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 
python3 main.py --anormly_ratio 1.0 --num_epochs 100  --batch_size 32  --mode memory_initial --dataset PSM  --data_path ./data/PSM/PSM/  --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train
# test
python3 main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 32  --mode test --dataset PSM  --data_path ./data/PSM/PSM/  --input_c 25 --output_c 25 --n_memory 10 --memory_initial False --phase_type test


# WACA
python3 main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 32  --mode train --dataset WACA  --data_path ./data/WACA/WACA/  --input_c 4 --output_c 4 --n_memory 10 --memory_initial False --win_size 1000 --phase_type None

python3 main.py --anormly_ratio 1.0 --num_epochs 100  --batch_size 32  --mode memory_initial --dataset WACA  --data_path ./data/WACA/WACA/  --input_c 4 --output_c 4 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --win_size 1000 --phase_type second_train

python3 main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 32  --mode test --dataset WACA  --data_path ./data/WACA/WACA/  --input_c 4 --output_c 4 --n_memory 10 --memory_initial False --phase_type test

# Window Test If user is an imposter
python3 main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 32  --mode window_test --dataset WACA  --data_path ./data/WACA/WACA/  --input_c 4 --output_c 4 --n_memory 10 --memory_initial False --phase_type test --threshold 3e20 --impostor


# # MSL

# python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 256 --mode train --dataset MSL --data_path ./data/MSL/MSL/ --input_c 55 --output_c 55 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anormly_ratio 1.0 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset MSL  --data_path ./data/MSL/MSL/  --input_c 55 --output_c 55 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 256  --mode test --dataset MSL  --data_path ./data/MSL/MSL/  --input_c 55 --output_c 55 --n_memory 10 --memory_initial False --phase_type test


# # SWaT

# python main.py --anormly_ratio 0.1 --num_epochs 100 --batch_size 256 --mode train --dataset SWaT --data_path ./data/SWaT/SWaT/ --input_c 51 --output_c 51 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anormly_ratio 0.1 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset SWaT  --data_path ./data/SWaT/SWaT/  --input_c 51 --output_c 51 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anormly_ratio 0.1 --num_epochs 100   --batch_size 256  --mode test --dataset SWaT  --data_path ./data/SWaT/SWaT/  --input_c 51 --output_c 51 --n_memory 10 --memory_initial False --phase_type test