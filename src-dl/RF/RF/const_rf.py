from os.path import join, abspath, dirname, pardir

# Base directory of your project
BASE_DIR = abspath(join(dirname(__file__), pardir))

# Where to save processed datasets
output_dir = join(BASE_DIR, 'RF/dataset/')

# Separator in your trace files (matches your trace formatting)
split_mark = '\t' 

# Closed-world setting
OPEN_WORLD = False

# Number of monitored sites (Tranco top-1000 websites)
MONITORED_SITE_NUM = 1000
MONITORED_INST_NUM = 15   # TODO: X samples per site

# No unmonitored sites in closed-world
UNMONITORED_SITE_NUM = 0
UNMONITORED_SITE_TRAINING = 0

# Paths for pretrained models (if used)
model_path = join(BASE_DIR, 'RF/pretrained/')

# Number of classes
num_classes = MONITORED_SITE_NUM
num_classes_ow = MONITORED_SITE_NUM + 1  # only used if open-world

# Maximum length for TAM (Traffic Analysis Matrix)
max_matrix_len = 1800

# Maximum page load time
maximum_load_time = 80

# Maximum number of packets to read from each trace
max_trace_length = 5000
