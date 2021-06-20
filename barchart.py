import pandas as pd
import time
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# font and axes style
fig = plt.figure(figsize=(12, 8))
fig.patch.set_facecolor('black')
ax = fig.add_subplot(1, 1, 1) # nrows, ncols, index
ax.patch.set_facecolor('black')
ax.spines['left'].set_color('w')
plt.xlim([0, 1.2])
plt.rcParams['axes.edgecolor']='#000000'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#000000'
plt.rcParams['ytick.color']='#ffffff'

JSON_filename = 'emotion-output.json'

def main():
	draw_chart()

# read json file of emotion output
def readJSON():
	with open(JSON_filename) as data_file:
		file_contents = json.load(data_file)
		return file_contents

# create y axes list
def make_key_list(data):
	keys = list(data.keys()) # ['Happy', 'Sad', 'Stressed', 'Relaxed']
	return keys

# create x axes list
def make_value_list(data):
	values = list(data.values()) # [1, 20, 3]
	return values

# draw bar chart initialy
def draw_chart():
	print("edit emotion-output.json to see the chart update")
	keys = make_key_list(readJSON()) # ['Happy', 'Sad', 'Stressed', 'Relaxed']
	values = [0,0,0,0] # [1, 20, 3]
	title_obj = plt.title('Emotion Recognition', fontsize=20)
	plt.setp(title_obj, color='w')
	barcollection = plt.barh(keys, values, color='w')
	# update chart every 1000ms
	anim = animation.FuncAnimation(fig, update_chart, interval=1000)
	plt.show()	

# update chart every 1000ms
# will display any changes to emotion-output.json
def update_chart(i):
    keys = make_key_list(readJSON())
    values = make_value_list(readJSON())
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    ax.patch.set_facecolor('black')
    ax.spines['left'].set_color('w')
    ax.yaxis.set_tick_params(labelsize=14)
    title_obj = plt.title('Emotion Recognition', fontsize=20)
    plt.setp(title_obj, color='w')
    plt.xlim([0, 5])
    barcollection = plt.barh(keys, values, color='w')
    for i, b in enumerate(barcollection):
        b.set_width(values[i])
    

if __name__ == "__main__":
    main()