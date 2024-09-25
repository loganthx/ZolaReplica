import os

def get_samples(filepaths):
	samples = []
	for filepath in filepaths:
		if '8cb' in filepath.lower():
			lbl = 0.0 
		elif 'e7' in filepath.lower():
			lbl = 1.0
		samples.append([filepath, lbl])
	return samples

def get_filepaths(path):
	filepaths = []
	for root, dirs, files in os.walk(path):
		for file in files:
			filepaths.append(f'{root}\\{file}')
	return filepaths

def get_config(path='data'):
	config = {
		'path': path,
		'samples': get_samples(get_filepaths(path)),
		'img_shape': (3, 100, 100),
		'batch_size': 16,
		'lr': 1e-3,
		'lr_step_size': 5, 
		'lr_gamma': 0.8,
		'epochs': 3,
		'split': 0.75, 
		'device': 'cuda',
		'loss': 'MSELoss',
			}
	return config


if __name__ == "__main__":
	config = get_config()












