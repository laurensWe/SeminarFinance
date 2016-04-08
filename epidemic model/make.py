for i in range(10):
    with open('draai {} tot {}.bat'.format(5*i,5*i+5),'w') as f:
        print('python "..\epidemicModel.py" {} {}'.format(5*i,5*i+5),file=f)
        print('PAUSE',file=f)