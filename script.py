import os
import make_data
import vis_data_distribution

os.system('python main.py -ns=1 -np=11 -d=True -op=H -pp=L -cp=H')
os.system('python main.py -ns=1 -np=11 -d=True -op=H -pp=H -cp=H')
os.system('python main.py -ns=1 -np=11 -d=True -op=L -pp=L -cp=H')
os.system('python main.py -ns=1 -np=11 -d=True -op=H -pp=L -cp=L')
os.system('python main.py -ns=1 -np=11 -d=True -op=L -pp=L -cp=L')
os.system('python main.py -ns=1 -np=11 -d=True -op=H -pp=H -cp=L')
os.system('python main.py -ns=1 -np=11 -d=True -op=L -pp=H -cp=H')
