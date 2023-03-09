import os
import subprocess
os.getcwd()
os.chdir('js1_code')
print("start_js1")
print("js1_maining")
cmd = "python main_js1.py"
runcmd = subprocess.call(cmd.split())

print("js1_featureing")
cmd = "python feature_extra.py"
runcmd = subprocess.call(cmd.split())

print("js1_testing")
cmd = "python test.py"
runcmd = subprocess.call(cmd.split())

print("js1_figing")
cmd = "python fig.py"
runcmd = subprocess.call(cmd.split())

print("js1_plotting")
cmd = "python auto_plot_js1.py"
runcmd = subprocess.call(cmd.split())
