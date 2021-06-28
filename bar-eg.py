import matplotlib.pyplot as plt
import matplotlib.animation as animation

JSON_filename = 'emotion-output.json'

print(range(1,8))

def barlist(n): 
    return [1/float(n*k) for k in range(1,8)]

# read json file of emotion output
def readJSON():
	with open(JSON_filename) as data_file:
		file_contents = json.load(data_file)
		return file_contents

fig=plt.figure()

n = 100 #Number of frames
x = range(1,8)
barcollection = plt.bar(x,barlist(1))

def animate(i):
    y = barlist(i+1)
    for i, b in enumerate(barcollection):
        b.set_height(y[i])

anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=n,
                             interval=100)

plt.show()

