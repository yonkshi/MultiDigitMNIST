standard:
	python3 generator.py --num_image_per_set 15000 --multimnist_path ./dataset/double_mnist --max_num_digit 8 --image_size 128 128

large_digit:
	python3 generator.py --num_image_per_set 15000 --multimnist_path ./dataset/double_mnist --max_num_digit 8 --image_size 128 128 --digit_size 28 28

local:
	python3 generator.py --num_image_per_set 320 --multimnist_path ./dataset/double_mnist --max_num_digit 8 --image_size 128 128