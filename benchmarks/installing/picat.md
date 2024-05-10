# Picat Solver

## Installing

```
cd deps
wget http://picat-lang.org/download/picat365_linux64.tar.gz # or any version more recent
tar -xf picat365_linux64.tar.gz
rm picat365_linux64.tar.gz
mv Picat picat
cd picat
git clone git@github.com:nfzhou/fzn_picat.git
```

We now add a script `fzn_picat.sh` to run the solver (where you replace `PATH_HOME` with the output of `echo $HOME`):

```
#!/bin/sh

PATH_HOME/deps/picat/picat PATH_HOME/deps/picat/fzn_picat/fzn_picat_sat "$@"
```

Make the script executable using:
```
chmod +x fzn_picat.sh
```

And we add the MiniZinc solver configuration file for Picat, first:

```
cp fzn_picat/picat.msc.in ~/.minizinc/solvers/picat.msc
```
Then edit the file with:
```
"version": "3.6.0",
"mznlib": "PATH_HOME/deps/picat/fzn_picat/mznlib",
"executable": "PATH_HOME/deps/picat/fzn_picat.sh",
```

## Running

You can check whether it works by running:
```
module load compiler/GCCcore/10.2.0 # for minizinc
cd ~/lattice-land/bench/benchmarks/data/mzn-challenge/2023/mrcpsp/
minizinc --solver org.picat-lang.picat -t 60000 mrcpsp.mzn j30_25_5.dzn
```
