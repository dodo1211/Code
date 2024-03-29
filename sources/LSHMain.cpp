/*
 * Copyright (c) 2004-2005 Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * MIT grants permission to use, copy, modify, and distribute this software and
 * its documentation for NON-COMMERCIAL purposes and without fee, provided that
 * this copyright notice appears in all copies.
 *
 * MIT provides this software "as is," without representations or warranties of
 * any kind, either expressed or implied, including but not limited to the
 * implied warranties of merchantability, fitness for a particular purpose, and
 * noninfringement.  MIT shall not be liable for any damages arising from any
 * use of this software.
 *
 * Author: Alexandr Andoni (andoni@mit.edu), Piotr Indyk (indyk@mit.edu)
 */

/*
  The main entry file containing the main() function. The main()
  function parses the command line parameters and depending on them
  calls the correspondin functions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include "headers.h"

#define N_SAMPLE_QUERY_POINTS 100

// The data set containing all the points.
PPointT *dataSetPoints = NULL;
// Number of points in the data set.
IntT nPoints = 0;
// The dimension of the points.
IntT pointsDimension = 0;
// The value of parameter R (a near neighbor of a point <q> is any
// point <p> from the data set that is the within distance
// <thresholdR>).
//RealT thresholdR = 1.0;

// The succes probability of each point (each near neighbor is
// reported by the algorithm with probability <successProbability>).
RealT successProbability = 0.9;

// Same as <thresholdR>, only an array of R's (for the case when
// multiple R's are specified).
RealT *listOfRadii = NULL;
IntT nRadii = 0;

RealT *memRatiosForNNStructs = NULL;

char sBuffer[600000];

/*
  Prints the usage of the LSHMain.
 */
void usage(char *programName){
  printf("Usage: %s #pts_in_data_set #queries dimension successProbability radius data_set_file query_points_file max_available_memory [-c|-p params_file]\n", programName);
}

inline PPointT readPoint(FILE *fileHandle){
  PPointT p;
  RealT sqrLength = 0;
  FAILIF(NULL == (p = (PPointT)MALLOC(sizeof(PointT))));
  FAILIF(NULL == (p->coordinates = (RealT*)MALLOC(pointsDimension * sizeof(RealT))));
  for(IntT d = 0; d < pointsDimension; d++){
    FSCANF_REAL(fileHandle, &(p->coordinates[d]));
    sqrLength += SQR(p->coordinates[d]);
  }
  fscanf(fileHandle, "%[^\n]", sBuffer);   //遇到\n就不读取了
  p->index = -1;
  p->sqrLength = sqrLength;
  return p;
}

// Reads in the data set points from <filename> in the array
// <dataSetPoints>. Each point get a unique number in the field
// <index> to be easily indentifiable.
void readDataSetFromFile(char *filename){
  FILE *f = fopen(filename, "rt");
  FAILIF(f == NULL);
  
  //fscanf(f, "%d %d ", &nPoints, &pointsDimension);
  //FSCANF_DOUBLE(f, &thresholdR);
  //FSCANF_DOUBLE(f, &successProbability);
  //fscanf(f, "\n");
  FAILIF(NULL == (dataSetPoints = (PPointT*)MALLOC(nPoints * sizeof(PPointT))));
  for(IntT i = 0; i < nPoints; i++){
    dataSetPoints[i] = readPoint(f);
    dataSetPoints[i]->index = i;
  }
}

// Tranforming <memRatiosForNNStructs> from
// <memRatiosForNNStructs[i]=ratio of mem/total mem> to
// <memRatiosForNNStructs[i]=ratio of mem/mem left for structs i,i+1,...>.
void transformMemRatios(){    //令每个半径所需的内存设为1，计算每个半径的存储结构占总内存的比例
  RealT sum = 0;
  for(IntT i = nRadii - 1; i >= 0; i--){
    sum += memRatiosForNNStructs[i];
    memRatiosForNNStructs[i] = memRatiosForNNStructs[i] / sum;
    //DPRINTF("%0.6lf\n", memRatiosForNNStructs[i]);
  }
  ASSERT(sum <= 1.000001);
}


int compareInt32T(const void *a, const void *b){  //按照由大到小的顺序进行排序
  Int32T *x = (Int32T*)a;
  Int32T *y = (Int32T*)b;
  return (*x > *y) - (*x < *y);
}

/*
  The main entry to LSH package. Depending on the command line
  parameters, the function computes the R-NN data structure optimal
  parameters and/or construct the R-NN data structure and runs the
  queries on the data structure.
 */
int main(int nargs, char **args){
  if(nargs < 9){
    usage(args[0]);
    exit(1);
  }

  //initializeLSHGlobal();

  // Parse part of the command-line parameters.
  nPoints = atoi(args[1]);
  IntT nQueries = atoi(args[2]);
  pointsDimension = atoi(args[3]);
  successProbability = atof(args[4]);
  char* endPtr[1];
  RealT thresholdR = strtod(args[5], endPtr);  //strtod将字符串转换成浮点数   //r=0.6
  //strtod()会扫描参数nptr字符串，跳过前面的空格字符，直到遇上数字或正负符号才开始做转换
  //，到出现非数字或字符串结束时('')才结束转换， 并将结果返回。
  //若endptr不为NULL，则会将遇到不合条件而终止的nptr中的字符指针由endptr传回。
  if (thresholdR == 0 || endPtr[1] == args[5]){   //确保阈值合法
    // The value for R is not specified, instead there is a file
    // specifying multiple R's.
    thresholdR = 0;

    // Read in the file
    FILE *radiiFile = fopen(args[5], "rt");
    FAILIF(radiiFile == NULL);
    fscanf(radiiFile, "%d\n", &nRadii);
    ASSERT(nRadii > 0);
    FAILIF(NULL == (listOfRadii = (RealT*)MALLOC(nRadii * sizeof(RealT))));
    FAILIF(NULL == (memRatiosForNNStructs = (RealT*)MALLOC(nRadii * sizeof(RealT))));
    for(IntT i = 0; i < nRadii; i++){
      FSCANF_REAL(radiiFile, &listOfRadii[i]);
      ASSERT(listOfRadii[i] > 0);
      FSCANF_REAL(radiiFile, &memRatiosForNNStructs[i]);
      ASSERT(memRatiosForNNStructs[i] > 0);
    }
  }else{
    nRadii = 1;     //半径的个数为1个
    FAILIF(NULL == (listOfRadii = (RealT*)MALLOC(nRadii * sizeof(RealT))));
    FAILIF(NULL == (memRatiosForNNStructs = (RealT*)MALLOC(nRadii * sizeof(RealT))));
    listOfRadii[0] = thresholdR;
    memRatiosForNNStructs[0] = 1;
  }
  DPRINTF("No. radii: %d\n", nRadii);
  //thresholdR = atof(args[5]);
  availableTotalMemory = atoll(args[8]);

  if (nPoints > MAX_N_POINTS) {
    printf("Error: the structure supports at most %d points (%d were specified).\n", MAX_N_POINTS, nPoints);
    fprintf(ERROR_OUTPUT, "Error: the structure supports at most %d points (%d were specified).\n", MAX_N_POINTS, nPoints);
    exit(1);
  }

  readDataSetFromFile(args[6]);    //数据集的文件名
  DPRINTF("Allocated memory (after reading data set): %lld\n", totalAllocatedMemory);

  Int32T nSampleQueries = N_SAMPLE_QUERY_POINTS;   //样本查询点的个数，100
  PPointT sampleQueries[nSampleQueries];      //对查询点编号
  Int32T sampleQBoundaryIndeces[nSampleQueries];   //第一个大于半径的点的编号，如果有多个半径的话，就会记录更多
  if ((nargs < 9) || (strcmp("-c", args[9]) == 0)){       //计算最优参数
    // In this cases, we need to generate a sample query set for
    // computing the optimal parameters.

    // Generate a sample query set.
    FILE *queryFile = fopen(args[7], "rt");              //打开查询集，以只读文本方式打开
    if (strcmp(args[7], ".") == 0 || queryFile == NULL || nQueries <= 0){
      // Choose several data set points for the sample query points.  //如果没有查询点就随机选择几个数据集点作为查询点
      for(IntT i = 0; i < nSampleQueries; i++){
	sampleQueries[i] = dataSetPoints[genRandomInt(0, nPoints - 1)];
      }
    }else{
      // Choose several actual query points for the sample query points.
      nSampleQueries = MIN(nSampleQueries, nQueries);    //MIN（100，9）
      Int32T sampleIndeces[nSampleQueries];              //定义了一个查询点样本索引数组
      for(IntT i = 0; i < nSampleQueries; i++){          
	  ////为什么要对查询点索引进行随机变化？ 想把样本查询点控制在一定的范围内，如果查询点过多，则样本点最多取100个查询点。
	      sampleIndeces[i] = genRandomInt(0, nQueries - 1);  //对查询点做了一下顺序的变化，对查询点的索引做随机处理。
      }
	   // 根据你给的比较条件进行快速排序，通过指针的移动实验排序，排序之后的结果仍然放在原数组中，必须自己写一个比较函数
	  //http://www.slyar.com/blog/stdlib-qsort.html qsort(数组起始地址，数组元素大小，每个元素的大小，函数指针指向比较函数)
      qsort(sampleIndeces, nSampleQueries, sizeof(*sampleIndeces), compareInt32T); //qsort，C语言标准库函数，对样本查询点的索引值进行排序
      //printIntVector("sampleIndeces: ", nSampleQueries, sampleIndeces);
      Int32T j = 0;
      for(Int32T i = 0; i < nQueries; i++){
	if (i == sampleIndeces[j]){  //如果样本查询点的索引值与实际查询点的索引值一致，读入点
	  sampleQueries[j] = readPoint(queryFile);
	  j++;
	  while (i == sampleIndeces[j]){   //如果样本查询点之后的索引值与实践查询点的索引值一致，则直接将此点的值赋给后面一点的值
	    sampleQueries[j] = sampleQueries[j - 1];   //覆盖之后索引点的值
	    j++;          //取后面的点
	  }
	}else{
	  fscanf(queryFile, "%[^\n]", sBuffer);
	  fscanf(queryFile, "\n");
	}
      }
      nSampleQueries = j;
      fclose(queryFile);
    }

    // Compute the array sampleQBoundaryIndeces that specifies how to
    // segregate the sample query points according to their distance
    // to NN.
	//边界sampleQBoundaryIndeces只会存取一个点的索引，该点的大小为第一个大于半径点的值
    sortQueryPointsByRadii(pointsDimension,
			   nSampleQueries,    //查询集的点的个数
			   sampleQueries,     //查询点的集合，函数运行完成后，点的值将以距离数据集合的距离由小到大的顺序排序
			   nPoints,           //数据集点的个数
			   dataSetPoints,     //数据集集合
			   nRadii,            //半径的个数
			   listOfRadii,        //半径的值
			   sampleQBoundaryIndeces);
  }
//之前的东西-c运行的，-p是不会运行的
  RNNParametersT *algParameters = NULL;
  PRNearNeighborStructT *nnStructs = NULL;
  if (nargs > 9) {
    // Additional command-line parameter is specified.
    if (strcmp("-c", args[9]) == 0) {
      // Only compute the R-NN DS parameters and output them to stdout. // 如果是-c，就只计算数据集参数，然后输出
      
      printf("%d\n", nRadii);           //打印出半径的个数：1个。 将写入到参数文件中，
      transformMemRatios();        //memRatiosForNNstructs,转换内存使用率。假设每个结构为1，每个半径占用的总内存的比率，用于计算内存
      for(IntT i = 0; i < nRadii; i++){   //看使用哪个样本查询点
	// which sample queries to use
	Int32T segregatedQStart = (i == 0) ? 0 : sampleQBoundaryIndeces[i - 1];   //起始点的位置
	Int32T segregatedQNumber = nSampleQueries - segregatedQStart;              //查询点的个数
	if (segregatedQNumber == 0) {                        //如果计算所得点的个数为0，就查询所有的点，从0到最后
	  // XXX: not the right answer
	  segregatedQNumber = nSampleQueries;
	  segregatedQStart = 0;
	}
	ASSERT(segregatedQStart < nSampleQueries);
	ASSERT(segregatedQStart >= 0);
	ASSERT(segregatedQStart + segregatedQNumber <= nSampleQueries);
	ASSERT(segregatedQNumber >= 0);
	RNNParametersT optParameters = computeOptimalParameters(listOfRadii[i],    //计算最优的运行时间，
								successProbability,
								nPoints,
								pointsDimension,
								dataSetPoints,
								segregatedQNumber,
								sampleQueries + segregatedQStart,
								(MemVarT)((availableTotalMemory - totalAllocatedMemory) * memRatiosForNNStructs[i])); //比率
								////memRatioForNNStructs[i]：近邻结构体每个半径所占用的内存比率，计算能用多少内存
	printRNNParameters(stdout, optParameters);  //将参数打印出来
      }
      exit(0);
    } else if (strcmp("-p", args[9]) == 0) {
      // Read the R-NN DS parameters from the given file and run the
      // queries on the constructed data structure.
      if (nargs < 10){
	usage(args[0]);
	exit(1);
      }
      FILE *pFile = fopen(args[10], "rt");    //读取参数文件，由lsh_computeParas产生
      FAILIFWR(pFile == NULL, "Could not open the params file.");
      fscanf(pFile, "%d\n", &nRadii);    //这里只取了参数文件中的半径，那参数文件中的其他数据怎样被取用的？？
     DPRINTF1("Using the following R-NN DS parameters:\n");   //使用R-NN DS(DateSet)参数
      DPRINTF("N radii = %d\n", nRadii);     //不知道将数据输出到哪里了？？
	 // printf("Using the following R-NN DS parameters:\n");
	 // printf("N radii=%d\n",nRadii);
      FAILIF(NULL == (nnStructs = (PRNearNeighborStructT*)MALLOC(nRadii * sizeof(PRNearNeighborStructT))));
      FAILIF(NULL == (algParameters = (RNNParametersT*)MALLOC(nRadii * sizeof(RNNParametersT))));
      for(IntT i = 0; i < nRadii; i++){
	        algParameters[i] = readRNNParameters(pFile);      //将参数信息，输出到屏幕上
  //	printRNNParameters(stderr, algParameters[i]);@727
      //printRNNParameters(stdout,algParameters[i]);
	        nnStructs[i] = initLSH_WithDataSet(algParameters[i], nPoints, dataSetPoints);  //根据用户输入的参数，初始化结构
      }

      pointsDimension = algParameters[0].dimension;
      FREE(listOfRadii);
      FAILIF(NULL == (listOfRadii = (RealT*)MALLOC(nRadii * sizeof(RealT))));
      for(IntT i = 0; i < nRadii; i++){
	listOfRadii[i] = algParameters[i].parameterR;
      }
    } else{
      // Wrong option.
      usage(args[0]);
      exit(1);
    }
  } else {
    FAILIF(NULL == (nnStructs = (PRNearNeighborStructT*)MALLOC(nRadii * sizeof(PRNearNeighborStructT))));
    // Determine the R-NN DS parameters, construct the DS and run the queries.
    transformMemRatios();
    for(IntT i = 0; i < nRadii; i++){
      // XXX: segregate the sample queries...
      nnStructs[i] = initSelfTunedRNearNeighborWithDataSet(listOfRadii[i], 
							   successProbability, 
							   nPoints, 
							   pointsDimension, 
							   dataSetPoints, 
							   nSampleQueries, 
							   sampleQueries, 
							   (MemVarT)((availableTotalMemory - totalAllocatedMemory) * memRatiosForNNStructs[i]));
    }
  }

 // DPRINTF1("X\n");@
  printf("X\n");

  IntT resultSize = nPoints;
  PPointT *result = (PPointT*)MALLOC(resultSize * sizeof(*result));
  PPointT queryPoint;
  FAILIF(NULL == (queryPoint = (PPointT)MALLOC(sizeof(PointT))));
  FAILIF(NULL == (queryPoint->coordinates = (RealT*)MALLOC(pointsDimension * sizeof(RealT))));

  FILE *queryFile = fopen(args[7], "rt");
  FAILIF(queryFile == NULL);
  TimeVarT meanQueryTime = 0;
  PPointAndRealTStructT *distToNN = NULL;
  for(IntT i = 0; i < nQueries; i++){

    RealT sqrLength = 0;
    // read in the query point.
    for(IntT d = 0; d < pointsDimension; d++){
      FSCANF_REAL(queryFile, &(queryPoint->coordinates[d]));
      sqrLength += SQR(queryPoint->coordinates[d]);   //向量到原点的距离
    }
    queryPoint->sqrLength = sqrLength;
    //printRealVector("Query: ", pointsDimension, queryPoint->coordinates);

    // get the near neighbors.
    IntT nNNs = 0;
    for(IntT r = 0; r < nRadii; r++){
      nNNs = getRNearNeighbors(nnStructs[r], queryPoint, result, resultSize);
      printf("Total time for R-NN query at radius %0.6lf (radius no. %d):\t%0.6lf\n", (double)(listOfRadii[r]), r, timeRNNQuery);
      meanQueryTime += timeRNNQuery;

      if (nNNs > 0){
	printf("Query point %d: found %d NNs at distance %0.6lf (%dth radius). First %d NNs are:\n", i, nNNs, (double)(listOfRadii[r]), r, MIN(nNNs, MAX_REPORTED_POINTS));
	
	// compute the distances to the found NN, and sort according to the distance
	FAILIF(NULL == (distToNN = (PPointAndRealTStructT*)REALLOC(distToNN, nNNs * sizeof(*distToNN))));
	for(IntT p = 0; p < nNNs; p++){
	  distToNN[p].ppoint = result[p];
	  distToNN[p].real = distance(pointsDimension, queryPoint, result[p]);
	}
	qsort(distToNN, nNNs, sizeof(*distToNN), comparePPointAndRealTStructT);  //C语言标准的函数

	// Print the points
	for(IntT j = 0; j < MIN(nNNs, MAX_REPORTED_POINTS); j++){
	  ASSERT(distToNN[j].ppoint != NULL);
	  printf("%09d\tDistance:%0.6lf\n", distToNN[j].ppoint->index, distToNN[j].real);   //打印点的坐标
	  CR_ASSERT(distToNN[j].real <= listOfRadii[r]);
	  //DPRINTF("Distance: %lf\n", distance(pointsDimension, queryPoint, result[j]));
	  //printRealVector("NN: ", pointsDimension, result[j]->coordinates);
	}
	break;
      }
    }
    if (nNNs == 0){
      printf("Query point %d: no NNs found.\n", i);
    }
  }
  if (nQueries > 0){
    meanQueryTime = meanQueryTime / nQueries;
    printf("Mean query time: %0.6lf\n", (double)meanQueryTime);
  }

  for(IntT i = 0; i < nRadii; i++){
    freePRNearNeighborStruct(nnStructs[i]);
  }
  // XXX: should ideally free the other stuff as well.


  return 0;
}
