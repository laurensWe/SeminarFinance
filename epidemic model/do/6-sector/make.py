f = 15
for i in range(11):
    with open('draai {} tot {}.bat'.format(f*i,f*i+f),'w') as file:
        print('python "..\epidemicModel.py" {} {}'.format(f*i,f*i+f),file=file)
        print('PAUSE',file=file)