Project - Continual learning
Option - MNIST (METHOD : GRADIENT EPISODIC MEMORY - GEM)
Group - 8
Members - Shivakalyan Soundarathiagarajan and Kowsick Venkatachalapathi
Note - Experiments,improvement and Learning in this project are based on the paper http://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf

Important Instructions
	•	The Python Notebook - CS677_Shiva_Kowsick_CL-GEM_MNIST.ipynb has all the  commands below - it will import all necessary modules, files accordingly
	•	Make sure to download both the Notebook CS677_Shiva_Kowsick_CL-GEM_MNIST.ipynb and the entire GIT hub project to get the dependent files.
  •	Results of GEM and all other models with different parameters are as follows

Results :

Version 	 location 	 method 	 dataset 	    learning rate	batch size 	time taken
=========================================================================================================================
V1 	 /results/results v1 	 single 	 mnist rotations 	0.003		10	 	2m
			 			 mnist permutations 	0.03		10	 	2m 30s
			 			 CIFAR_100 		1		10	 	3m
		 		 GEM 	 	 mnist permutations 	0.1		10	 	6m 30s
			 			 mnist permutations 	0.1		10	 	8m 30s
			 			 CIFAR_100 		0.1		10	 	3h 30m
V2 	 /results/results v2 	 single 	 mnist rotations 	0.001		10	 	2m
			 			 mnist permutations 	0.01		10	 	2m 30s
			 			 CIFAR_100 		0.1		10	 	3m
		 		 GEM 	 	 mnist permutations 	0.3		10	 	5m 30s
			 			 mnist permutations 	0.3		10	 	7sm 30s
			 			 CIFAR_100 		0.3		10	 	2h 45m

Config Options :

MNIST_ROTA="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_rotations.pt    --cuda no  --seed 0"
MNIST_PERM="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_permutations.pt --cuda no  --seed 0"
CIFAR_100i="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 2500 --data_file cifar100.pt           --cuda yes --seed 0"

# datasets
cd data/
cd raw/

!python raw.py

cd ..

!python mnist_permutations.py \
  --o mnist_permutations.pt \
  --seed 0 \
  --n_tasks 20

!python mnist_rotations.py \
  --o mnist_rotations.pt\
  --seed 0 \
  --min_rot 0 \
  --max_rot 180 \
  --n_tasks 20

!python cifar100.py \
  --o cifar100.pt \
  --seed 0 \
  --n_tasks 20

cd ..

# model "single"
!python main.py $MNIST_ROTA --model single --lr 0.003
!python main.py $MNIST_PERM --model single --lr 0.03
!python main.py $CIFAR_100i --model single --lr 1.0
 
# model "independent"
!python main.py $MNIST_ROTA --model independent --lr 0.1  --finetune yes 
!python main.py $MNIST_PERM --model independent --lr 0.03 --finetune yes 
!python main.py $CIFAR_100i --model independent --lr 0.3  --finetune yes 

# model "multimodal"
!python main.py $MNIST_ROTA  --model multimodal --lr 0.1
!python main.py $MNIST_PERM  --model multimodal --lr 0.1

# model "EWC"
!python main.py $MNIST_ROTA --model ewc --lr 0.01 --n_memories 1000 --memory_strength 1000
!python main.py $MNIST_PERM --model ewc --lr 0.1  --n_memories 10   --memory_strength 3
!python main.py $CIFAR_100i --model ewc --lr 1.0  --n_memories 10   --memory_strength 1


# model "GEM"
!python main.py $MNIST_ROTA --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5
!python main.py $MNIST_PERM --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5
!python main.py $CIFAR_100i --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5

# plot results
cd results/
!python plot_results.py
cd ..

