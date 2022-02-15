This directory stores information of correct patch's score ranking.
We discuss situation of ASE patches and ASE patches + prapr patches respectively and ASE patches + prapr patches + developer patches.
1. If all patches are overfitting, ranking is set as NAN
2. If all patches are correct, ranking is set as 1
3. If no ASE and prapr patches available, ignore this bug
4. If there are both correct and overfitting patches, ranking is the highest ranking of the correct patch