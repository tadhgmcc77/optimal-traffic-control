'''
import xml.etree.ElementTree as ET

# Open the XML file
with open('summary-heavy.xml', 'rt') as f:
    # Parse the XML file and get the root element
    tree = ET.parse(f)
    root = tree.getroot()

# Iterate through the child elements of the root and print their tag and text
for child in root:
    print(child.tag, child.text)
'''


import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Parse the XML file and get the root element
tree = ET.parse('BASIC-K-summary-low.xml')
root = tree.getroot()
waiting_times = []
times = []
# Get the value of the meanWaitingTime attribute from the step element

for step in root.findall('step'):
    mean_waiting_time = step.get('meanWaitingTime')
    time = step.get('time')
    counter = 0
    if mean_waiting_time == '0.00':
        waiting_times.append(float(mean_waiting_time))
    else:
        waiting_times.append(float(mean_waiting_time) + 0.03)
    times.append(time)
    counter += 1


tree2 = ET.parse('ADAPTIVE-K-summary-low.xml')
root2 = tree2.getroot()
waiting_times2 = []
times2 = []
# Get the value of the meanWaitingTime attribute from the step element

for step in root2.findall('step'):
    mean_waiting_time = step.get('meanWaitingTime')
    time = step.get('time')
    if mean_waiting_time == '0.00':
        waiting_times2.append(float(mean_waiting_time))
    else:
        waiting_times2.append(float(mean_waiting_time) - 0.03)
    times2.append(time)




tree3 = ET.parse('ACTUATED-K-summary-low.xml')
root3 = tree3.getroot()
waiting_times3 = []
times3 = []
# Get the value of the meanWaitingTime attribute from the step element

for step in root3.findall('step'):
    mean_waiting_time = step.get('meanWaitingTime')
    time = step.get('time')
    if mean_waiting_time == '0.00':
        waiting_times3.append(float(mean_waiting_time))
    else:
        waiting_times3.append(float(mean_waiting_time) + 0.03)
    times3.append(time)

    
tree4 = ET.parse('K-summary-low.xml')
root4 = tree4.getroot()
waiting_times4 = []
times4 = []
# Get the value of the meanWaitingTime attribute from the step element

for step in root4.findall('step'):
    mean_waiting_time = step.get('meanWaitingTime')
    time = step.get('time')
    if mean_waiting_time == '0.00':
        waiting_times4.append(float(mean_waiting_time))
    else:
        waiting_times4.append(float(mean_waiting_time) - 0.03)
    times4.append(time)





print(waiting_times[1200])
print(waiting_times2[1200])
print(waiting_times3[1200])




fig, ax = plt.subplots()


ax.set_xticks([300, 600, 900, 1200, 1500, 1800])
ax.set_xticklabels(['300', '600', '900', '1200', '1500', '1800'])

# set the y-axis tick locations and labels
#ax.set_yticks([1,2,4,6,8,10])
#ax.set_yticklabels(['1', '2', '4', '6', '8', '10'])



ax.plot(times, waiting_times, label='Basic')
ax.plot(times, waiting_times2, label='Adaptive')
ax.plot(times, waiting_times3, label='Actuated')
ax.plot(times, waiting_times4, label='No Rules')

ax.set_xlabel('time')
ax.set_ylabel('average mean waiting time')
ax.set_title('My Graph')



ax.legend()
plt.show()

#import all files at once
#add to list/dict and plot
