README
------

This directory contains the data for the [PARSEME 2.0 Multilingual Shared Task on Identification and Paraphrasing of Multiword Expressions](https://unidive.lisn.upsaclay.fr/doku.php?id=other-events:parseme-st). The corpora have been manually annotated for multiword expressions (MWEs) following the [PARSEME unified guidelines version 2.0](https://parsemefr.lis-lab.fr/parseme-st-guidelines/2.0/).

**Trial data:** subfolders `subtask1_trial` and `subtask2_trial` contain examples illustrating how the training/evaluation data will look like. All files have toy sizes (true files will be larger). Note that, while trial files are provided in English and French, English is not part of the shared task 2.0. Moreover, `dev` data for system development is not present in the trial folders, whereas it is provided for `subtask1`.

**Training data (subtask 1):** subfolder `subtask1` contains the training data and evaluation tools for subtask 1 (see details below). Since there is no training data for subtask 2, the folder `subtask2` contains only the evaluation tools for this subtask. For both subtasks, the folder `tools` contains the evaluation tools whereas folders named with the uppercase ISO language code contain the data for each of the languages (along with the READMEs and corpus statistics).

**Evaluation data:** the blind evaluation/test files are now available, in the corresponding subfolders, named `test.blind.cupt` for subtask 1 and `test.blind.json` for subtask 2.

### Subtask 1: MWE identification

This subtask is an extension of [PARSEME shared tasks](https://gitlab.com/parseme/corpora/-/wikis/home#shared-tasks) on automatic identification of verbal MWEs.
* **Task**: Given a raw text, automatically underline/tag MWEs in it.
* **Format**: All files are provided in [cupt format](https://gitlab.com/parseme/corpora/-/wikis/CUPT-format).
* **Trial data**: The example/trial data for this subtask is available in folder `subtask1_trial`. Trial files are provided in French (subfolder `FR`) and in English (subfolder `EN`).
  * `trial.train.cupt`: Example of a training corpus with manually annotated MWEs. Training data will be provided in advance to allow training/fine-tuning systems.
  * `trial.test.blind.cupt`: Example of a blind version of the test data; systems should take such a file on input and provide automatic predictions of MWEs in column 11. Blind test data will be provided at the beginning of the evaluation phase and system developers will have a few days to provide their predictions. 
  * `trial.test.system.cupt`: Sample system predictions of MWEs, this exemplifies what a system developers will need to submit at the end of the evaluation phase.
  * `trial.test.cupt`: Example of a gold annotation of the test data; system predictions will be scored based on comparison to this gold annotations; gold annotations will not be available before the end of the shared task evaluation phase.
  * `*-stats.md`: Statistics of the corpora - size and MWE category distribution.
* **Language data**: The actual training data is available in folder `subtask1`. Training data are provided in Egyptian (`EGY`), Farsi (`FA`), Ancient Greek (`GRC`), Hebrew (`HE`), Japanese (`JA`), Georgian (`KA`), Latvian (`LV`), Dutch (`NL`), Polish (`PL`), Brazilian Portuguese (`PT`), Romanian (`RO`), Slovene (`SL`), Serbian (`SR`), and Swedish (`SV`). Data for Modern Greek (`EL`), French (`FR`) and Ukrainian (`UK`) will be release by the end of October 2025.  
  * `train.cupt`: training corpus with manually annotated MWEs, provided in advance to allow training/fine-tuning systems.
  * `dev.cupt`: development corpus with manually annotated MWEs, provided in advance to allow evaluating systems on held-out data before the test corpus is available.
  * `test.blind.cupt`: systems should take such a file on input and provide automatic predictions of MWEs in column 11. Blind test data will be provided at the beginning of the evaluation phase and system developers will have a few days to provide their predictions. 
  * `README.md`: documentation of the corpus and annotations provided by the language team.
  * `*-stats.md`: Statistics of the corpora - size and MWE category distribution.
* **Evaluation tools**: The evaluation script and related tools are contained in `subtask1/tools`. The folder contains a `README.md` describing the scripts and evaluation metrics.  
  
### Subtask 2: MWE paraphrasing

This subtask is a new task proposed for PARSEME 2.0, consisting of identifying and reformulating Multiword Expressions to remove them from a text.
* **Task**: Given a raw text, automatically identify and reformulate the MWE. Exactly one MWE is in each text, of VID, NID or AdjID type.
* **Trial data**: The example data for this subtask is available in folder `subtask2_trial`. Trial files are provided in French (subfolder `FR`) and in English (subfolder `EN`). All files are provided in a JSON format, described in the [wiki](https://gitlab.com/parseme/corpora/-/wikis/mwe-paraphrasing-data).
* `gold.json`: Example of a gold file with the original sentence, manual annotations, and some metadata. This file will be used to evaluate the prediction file.
* `blind.json`: Example of a blind version of the test data; systems should take such a file on input and provide automatic rephrasing. Blind test data will be provided at the beginning of the evaluation phase and system developers will have a few days to provide their predictions.
* `prediction.json`: Sample system predictions of rephrasings, this exemplifies what a system developers will need to submit at the end of the evaluation phase.
