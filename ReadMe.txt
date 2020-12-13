Abstract—Edge detection is important in many computer vision applications. Traditional methods such as canny edge detection etc are limited in nature as they are hand crafted. In this paper, a flexible and powerful approach for detecting edges with cellular automata using genetic algorithms is explored. The implementation Is covered along with an evaluation on a few test images. Quality metric PSNR and edge detection performance metric MS-SSIM (Multi Scale Structural Similarity Index) are used to test the results obtained. A subjective assessment is also made to discuss the visual aspects of the test images.
There are three kinds of files: 

1)final_project_main_multiple_instances_versions.py
	This contains all the instances and versions discussed in the paper and takes approx 10-12 hrs to execute.(each instance has 2 	runs. runs>1 is must)

	Note:- use calculatefitness_2_parallel module instead of calculatefitness_2 for faster execution if the system is multicore. Joblib is the module responsible for this performance. If a brokenpool error is thrown restart the python kernel and run again.

	Caution: Running this file will change the previous results in the 	folders GA version 1, GA version 2, GA version 3.

2)final_project_main_best_instance.py 
	This file contains the best instance and takes approx 1-2 hrs to 		execute for the given parameters. (there are 2 runs. runs>1 is 	must)

	Note:- the population and offspring size should be even otherwise 		an error is thrown by the crowding implementation and the runs 		must be greater than 1.

3)final_project_main_best_instance_testing.py
	This file contains the bonus. It performs edge detection using the 	rule table stored in BestRuleTable.json (obtained by running best 		instance) on the unseen images discussed in paper in the results 		section. The results are stored in folder Test Image Results.

To run the above files run the comand "python filename" and press enter in CMD.
