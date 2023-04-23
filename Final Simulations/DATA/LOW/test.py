'''
import xml.etree.ElementTree as ET

# Open the XML file
with open('summary-low.xml', 'rt') as f:
    # Parse the XML file and get the root element
    tree = ET.parse(f)
    root = tree.getroot()

# Iterate through the child elements of the root and print their tag and text
for child in root:
    print(child.tag, child.text)
'''


import xml.etree.ElementTree as ET

# Parse the XML file and get the root element
tree = ET.parse('ADAPTIVE-SI-summary-low.xml')
root = tree.getroot()

# Get the value of the meanWaitingTime attribute from the step element
counter = 0
for step in root.findall('step'):
    mean_waiting_time = step.get('meanWaitingTime')
    counter += 1
    print(counter, mean_waiting_time)
    if counter >= 500:
        break


