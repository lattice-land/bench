# ACE Solver

## Installing

```
cd deps
git clone https://github.com/xcsp3team/ACE.git
cd ACE
git checkout 2.3 # or any version more recent
```

Then we must install `gradle` and compile ACE:

```
salloc -p interactive --qos debug -C batch # or just "si" on uni.lu machines

cd ~
python -m venv eb-venv
source eb-venv/bin/activate
load_local_easybuild "2022b" # You must add the function `load_local_easybuild` to your .bashrc, see https://ulhpc-tutorials.readthedocs.io/en/latest/tools/easybuild/#install-a-missing-software-with-a-more-recent-toolchain
python3 -m pip install easybuild==4.7.1
eb -r Gradle-6.9.1.eb

export MODULEPATH=${MODULEPATH}:$HOME/.local/easybuild/aion/2022b/epyc/modules/all # on Aion, would be different on Iris.
module load devel/Gradle/6.9.1
module load lang/Java/11
cd deps/ACE
gradle build -x test
```

## Running

You can then test with:

```
module load lang/Java/11
cd ~/lattice-land/bench/benchmarks/data/mzn-challenge/2023/mrcpsp/ # in this repository
java -jar build/libs/ACE-2.3.jar
```
