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
  The self-tuning module. This file contains all the functions for
  estimating the running times and for computing the optimal (at least
  in estimation) parameters for the R-NN data structure (within the
  memory limits).
 */

#include "headers.h"

// Computes how much time it takes to run timing functions (functions
// that compute timings) -- we need to substract this value when we
// compute the length of an actual interval of time.
//计算函数运行的时间，我们需要减去这个值，得到精确的处理时间
void tuneTimeFunctions(){
  timevSpeed = 0;
  // Compute the time needed for a calls to TIMEV_START and TIMEV_END
  IntT nIterations = 100000;
  TimeVarT timeVar = 0;
  for(IntT i = 0; i < nIterations; i++){
    TIMEV_START(timeVar);
    TIMEV_END(timeVar);
  }
  timevSpeed = timeVar / nIterations;
  DPRINTF("Tuning: timevSpeed = %0.9lf\n", timevSpeed);
}

/* 
 * Given a set of queries, the data set, and a set of (sorted) radii,
 * this function will compute the <boundaryIndeces>, i.e., the indeces
 * at which the query points go from one "radius class" to another
 * "radius class".
 * 
 * More formally, the query set is sorted according to the distance to
 * NN. Then, function fills in <boundaryIndeces> such that if
 * <boundaryIndeces[i]=A>, then all query points with index <A have
 * their NN at distance <=radii[i], and other query points (with index
 * >=A) have their NN at distance >radii[i].
 *
 * <boundaryIndeces> must be preallocated for at least <nQueries>
 * elements.
 */
void sortQueryPointsByRadii(IntT dimension,
			    Int32T nQueries, 
			    PPointT *queries, 
			    Int32T nPoints, 
			    PPointT *dataSet,
			    IntT nRadii,
			    RealT *radii,
			    Int32T *boundaryIndeces){
  ASSERT(queries != NULL);
  ASSERT(dataSet != NULL);
  ASSERT(radii != NULL);
  ASSERT(boundaryIndeces != NULL);


  PPointAndRealTStructT *distToNN = NULL;  //存放查询点，(ppoint,real)  ppoint=(index,coordinates,sqrLength)
  FAILIF(NULL == (distToNN = (PPointAndRealTStructT*)MALLOC(nQueries * sizeof(*distToNN))));
  for(IntT i = 0; i < nQueries; i++){      //计算最小距离，计算每个查询点的最近距离，没有把数据集的距离计算进去 ，只计算了距离，没有存放真正的数据点
    distToNN[i].ppoint = queries[i];
    distToNN[i].real = distance(dimension, queries[i], dataSet[0]);
    for(IntT p = 0; p < nPoints; p++){      //查询集的一个点与数据集的每个点进行比较
      RealT dist = distance(dimension, queries[i], dataSet[p]);    //求得查询点与数据集点的距离
      if (dist < distToNN[i].real){               //将查询点与数据集点的最小距离存入distToNN[i].real  //注意，只存储了距离
	distToNN[i].real = dist;
      }
    }
  }

  qsort(distToNN, nQueries, sizeof(*distToNN), comparePPointAndRealTStructT);   //把数组按照查询点所对应的最近邻的距离做下排序。为了定半径。

  IntT radiusIndex = 0;
  for(IntT i = 0; i < nQueries; i++) {
    //DPRINTF("%0.6lf\n", distToNN[i].real);
    queries[i] = distToNN[i].ppoint; // copy the sorted queries array back to <queries>   //复制排序之后的数组到queries
    while ((distToNN[i].real > radii[radiusIndex]) && (radiusIndex < nRadii)) {           //每个半径只记录一个边界值
      boundaryIndeces[radiusIndex] = i;                                              //将第一个大于半径的点的索引编号存入boundary
      radiusIndex++;
    }
  }

  FREE(distToNN);
}

// Determines the run-time coefficients of the different parts of the  //确定查询算法不同部分的运行时间
// query algorithm. Values that are computed and returned are
// <lshPrecomp>, <uhashOver>, <distComp>. <lshPrecomp> is the time for
// pre-computing one function from the LSH family. <uhashOver> is the
// time for getting a bucket from a hash table (of buckets).<distComp>
// is the time to compute one distance between two points. These times
// are computed by constructing a R-NN DS on a sample data set and
// running a sample query set on it.
void determineRTCoefficients(RealT thresholdR, 
			     RealT successProbability, 
			     BooleanT useUfunctions, 
			     IntT typeHT, 
			     IntT dimension, 
			     Int32T nPoints, 
			     PPointT *realData, 
			     RealT &lshPrecomp, 
			     RealT &uhashOver, 
			     RealT &distComp){

  // use a subset of the original data set.   使用原始数据集的一个子集
  // there is not much theory behind the formula below.    //减小运算规模
  IntT n = nPoints / 50;    //最多生成n各点，缩小50倍
  if (n < 100) {            //如果生成的点的个数小于100，则使桶的数量与数据集点的数量一样多
    n = nPoints;
  }
  if (n > 10000) {
    n = 10000;
  }

  // Initialize the data set to use.
  PPointT *dataSet;
  FAILIF(NULL == (dataSet = (PPointT*)MALLOC(n * sizeof(PPointT))));
  for(IntT i = 0; i < n; i++){           //从真实数据集中随机取n个点 （最多10000个）
    dataSet[i] = realData[genRandomInt(0, nPoints - 1)];
  }

  IntT hashTableSize = n;                //哈希表大小也初始化为n
  RNNParametersT algParameters;
  algParameters.parameterR = thresholdR;   //半径
  algParameters.successProbability = successProbability;
  algParameters.dimension = dimension;
#ifdef USE_L1_DISTANCE
  algParameters.parameterR2 = thresholdR;       //使用L1距离，R2=R
#else  
  algParameters.parameterR2 = SQR(thresholdR);   //使用L2  R2=R^2
#endif
  algParameters.useUfunctions = useUfunctions;
  algParameters.parameterK = 16;       //k 设定为16，只是测试，估算运算时间，可能是先随机设置一个时间，之后再在代码中改成16，因为16是bestK.
  algParameters.parameterW = PARAMETER_W_DEFAULT;    //W=4，manuel中说经过多次测试，4是最好的值
  algParameters.parameterT = n;                     //点的个数
  algParameters.typeHT = typeHT;                      //桶的类型

  if (algParameters.useUfunctions){
    algParameters.parameterM = computeMForULSH(algParameters.parameterK, algParameters.successProbability);     //经过改进的L和M
    algParameters.parameterL = algParameters.parameterM * (algParameters.parameterM - 1) / 2;
  }else{
    algParameters.parameterM = computeLfromKP(algParameters.parameterK, algParameters.successProbability);          //论文里面的M=L 
    algParameters.parameterL = algParameters.parameterM;
  }

//   FAILIF(NULL == (dataSet = (PPointT*)MALLOC(n * sizeof(PPointT))));
//   for(IntT i = 0; i < n; i++){
//     FAILIF(NULL == (dataSet[i] = (PPointT)MALLOC(sizeof(PointT))));
//     FAILIF(NULL == (dataSet[i]->coordinates = (RealT*)MALLOC(dimension * sizeof(RealT))));

//     dataSet[i]->index = i;
//     sqrLength = 0;
//     for(IntT d = 0; d < dimension; d++){
//       if (i == 0) {
// 	dataSet[i]->coordinates[d] = genUniformRandom(-100, 100);
//       }else{
// 	dataSet[i]->coordinates[d] = dataSet[0]->coordinates[d];
//       }
//       sqrLength += SQR(dataSet[i]->coordinates[d]);
//     }
//     dataSet[i]->sqrLength = sqrLength;
//   }

  // switch on timing
  BooleanT tempTimingOn = timingOn;    //初始化为True
  timingOn = TRUE;

  // initialize result arrays
  PPointT *result = NULL;             //结果集以及其初始化
  IntT resultSize = 0;
  IntT nNNs;
  IntT nSucReps;

  do{
    // create the test structure
    PRNearNeighborStructT nnStruct;
    switch(algParameters.typeHT){
    case HT_LINKED_LIST:
      nnStruct = initLSH(algParameters, n);
      // add points to the test structure
      for(IntT i = 0; i < n; i++){
	addNewPointToPRNearNeighborStruct(nnStruct, realData[i]);
      }
      break;
    case HT_HYBRID_CHAINS:
      nnStruct = initLSH_WithDataSet(algParameters, n, dataSet);   //初始化数据结构，参数集，点的个数，数据集，对点进行映射转换，桶进行映射转换，点存入桶中
      break;
    default:
      ASSERT(FALSE);
    }

    // query point
    PPointT queryPoint;
//     FAILIF(NULL == (queryPoint = (PPointT)MALLOC(sizeof(PointT))));
//     FAILIF(NULL == (queryPoint->coordinates = (RealT*)MALLOC(dimension * sizeof(RealT))));
//     RealT sqrLength = 0;
//     for(IntT i = 0; i < dimension; i++){
//       queryPoint->coordinates[i] = dataSet[0]->coordinates[i];
//       //queryPoint->coordinates[i] = 0.1;
//       sqrLength += SQR(queryPoint->coordinates[i]);
//     }
    //queryPoint->coordinates[0] = dataPoint->coordinates[0] + 0.0001;
    //queryPoint->sqrLength = sqrLength;

    // reset the R parameter so that there are no NN neighbors.
    setResultReporting(nnStruct, FALSE);
    //DPRINTF1("X\n");

    lshPrecomp = 0;
    uhashOver = 0;
    distComp = 0;
    IntT nReps = 20;
    nSucReps = 0;
    for(IntT rep = 0; rep < nReps; rep++){
      queryPoint = realData[genRandomInt(0, nPoints - 1)];   //查询点为数据集中随机抽取出来的一个点
      timeComputeULSH = 0;
      timeGetBucket = 0;
      timeCycleBucket = 0;
      nOfDistComps = 0;                //点与点比较的次数
      nNNs = getNearNeighborsFromPRNearNeighborStruct(nnStruct, queryPoint, result, resultSize);   //返回一个数，用于检查是否运行正常
      //DPRINTF("Time to compute LSH: %0.6lf\n", timeComputeULSH);
      //DPRINTF("Time to get bucket: %0.6lf\n", timeGetBucket);
      //DPRINTF("Time to cycle through buckets: %0.9lf\n", timeCycleBucket);
      //DPRINTF("N of dist comp: %d\n", nOfDistComps);

      ASSERT(nNNs == 0);
      if (nOfDistComps >= MIN(n / 10, 100)){    //与足够的点比较过，才将时间计入
	nSucReps++;
	lshPrecomp += timeComputeULSH / algParameters.parameterK / algParameters.parameterM;  //每个点的一个维度存放入hashing桶的时间
	uhashOver += timeGetBucket / algParameters.parameterL;     //取得桶的时间
	distComp += timeCycleBucket / nOfDistComps;
      }
    }

    if (nSucReps >= 5){
      lshPrecomp /= nSucReps;
      uhashOver /= nSucReps;
      distComp /= nSucReps;
      DPRINTF1("RT coeffs computed.\n");
    }else{
      algParameters.parameterR *= 2; // double the radius and repeat  //比较的点数不够，将半径扩大，重复比较
      DPRINTF1("Could not determine the RT coeffs. Repeating.\n");
    }

    freePRNearNeighborStruct(nnStruct);

  }while(nSucReps < 5);       //做一个有效值的判断，要获得5次有效值

  FREE(dataSet);
  FREE(result);

  timingOn = tempTimingOn;
}

/*
  The function <p> from the paper (probability of collision of 2
  points for 1 LSH function).
 */
RealT computeFunctionP(RealT w, RealT c){   //L2 norm  //两个点在一个桶中冲突的可能性 p1
  RealT x = w / c;
  return 1 - ERFC(x / M_SQRT2) - M_2_SQRTPI / M_SQRT2 / x * (1 - EXP(-SQR(x) / 2));  //ERFC为余误差函数，M_SQRT2为math.h中的函数，值为pi
														//M_2_SQRTPI= 2/sqrt(pi) ,exp(x)=e^x 程序公式1
}

// Computes the parameter <L> of the algorithm, given the parameter
// <k> and the desired success probability
// <successProbability>. Functions <g> are considered all independent
// (original scheme).
IntT computeLfromKP(IntT k, RealT successProbability){
  return CEIL(LOG(1 - successProbability) / LOG(1 - POW(computeFunctionP(PARAMETER_W_DEFAULT, 1), k)));   //向上取整,LOG=ln ，公式6，返回L的值
}

// Computes the parameter <m> of the algorithm, given the parameter
// <k> and the desired success probability <successProbability>. Only
// meaningful when functions <g> are interdependent (pairs of
// functions <u>, where the <m> functions <u> are each k/2-tuples of
// independent LSH functions).    //g函数是相关时，有用！ 为了减少计算时间
IntT computeMForULSH(IntT k, RealT successProbability){
  ASSERT((k & 1) == 0); // k should be even in order to use ULSH.
  RealT mu = 1 - POW(computeFunctionP(PARAMETER_W_DEFAULT, 1), k / 2);    //1-p1^(k/2)   ，为1 说明c=1
  RealT P = successProbability;
  RealT d = (1-mu)/(1-P)*1/LOG(1/mu) * POW(mu, -1/(1-mu));     //(p1^k/2)/delta*log(1/(p1^k/2))*((1-p1^k/2)^(-1/(p1^(k/2)))) 
  RealT y = LOG(d);
  IntT m = CEIL(1 - y/LOG(mu) - 1/(1-mu));
  while (POW(mu, m-1) * (1 + m * (1-mu)) > 1 - P){      //13
    m++;
  }
  return m;
}
RealT estimateNCollisions(IntT nPoints, IntT dim, PPointT *dataSet, PPointT query, IntT k, IntT L, RealT R){
  RealT sumCollisions = 0;
  for(IntT i = 0; i < nPoints; i++){
    if (query != dataSet[i]) {
      RealT dist = distance(dim, query, dataSet[i]);     //计算在桶中查询点与数据集中点的距离
      sumCollisions += POW(computeFunctionP(PARAMETER_W_DEFAULT, dist / R), k);    // dist很有可能就是表示r2，dist越大算出来的概率越小。
    }
  }
  return L * sumCollisions;        //求冲突的期望值   //sumCollosions是在所有桶中的冲突点的数目
}

RealT estimateNCollisionsFromDSPoint(IntT nPoints, IntT dim, PPointT *dataSet, IntT queryIndex, IntT k, IntT L, RealT R){
  RealT sumCollisions = 0;
  for(IntT i = 0; i < nPoints; i++){
    if (queryIndex != i) {
      RealT dist = distance(dim, dataSet[queryIndex], dataSet[i]);
      sumCollisions += POW(computeFunctionP(PARAMETER_W_DEFAULT, dist / R), k);
    }
  }
  return L * sumCollisions;
}

RealT estimateNDistinctCollisions(IntT nPoints, IntT dim, PPointT *dataSet, PPointT query, BooleanT useUfunctions, IntT k, IntT LorM, RealT R){
  RealT sumCollisions = 0;
  for(IntT i = 0; i < nPoints; i++){
    if (query != dataSet[i]) {
      RealT dist = distance(dim, query, dataSet[i]);
      if (!useUfunctions){ 
	sumCollisions += 1-POW(1-POW(computeFunctionP(PARAMETER_W_DEFAULT, dist / R), k), LorM);        //1-（1-p1^k）^L，两点冲突的概率
      }else{
	RealT mu = 1 - POW(computeFunctionP(PARAMETER_W_DEFAULT, dist / R), k / 2);
	RealT x = POW(mu, LorM - 1);
	sumCollisions += 1 - mu * x - LorM * (1 - mu) * x;    //查询点与数据集点冲突的概率总和，为能查询到的近邻点的数目
      }
    }
  }
  return sumCollisions;
}

RealT estimateNDistinctCollisionsFromDSPoint(IntT nPoints, IntT dim, PPointT *dataSet, IntT queryIndex, BooleanT useUfunctions, IntT k, IntT LorM, RealT R){
  RealT sumCollisions = 0;
  for(IntT i = 0; i < nPoints; i++){
    if (queryIndex != i) {
      RealT dist = distance(dim, dataSet[queryIndex], dataSet[i]);
      if (!useUfunctions){
	sumCollisions += 1-POW(1-POW(computeFunctionP(PARAMETER_W_DEFAULT, dist / R), k), LorM);
      }else{
	RealT mu = 1 - POW(computeFunctionP(PARAMETER_W_DEFAULT, dist / R), k / 2);
	RealT x = POW(mu, LorM - 1);
	sumCollisions += 1 - mu * x - LorM * (1 - mu) * x;
      }
    }
  }
  return sumCollisions;
}

/*
  Given the actual data set <dataSet>, estimates the values for
  algorithm parameters that would give the optimal running time of a
  query. 

  The set <sampleQueries> is a set with query sample points
  (R-NN DS's parameters are optimized for query points from the set
  <sampleQueries>). <sampleQueries> could be a sample of points from the
  actual query set or from the data set. When computing the estimated
  number of collisions of a sample query point <q> with the data set
  points, if there is a point in the data set with the same pointer
  with <q> (that is when <q> is a data set point), then the
  corresponding point (<q>) is not considered in the data set (for the
  purpose of computing the respective #collisions estimation).
//如果查询样本点来自数据集，则为了考虑碰撞估计，不讲对应的样本查询点考虑进数据集合
  The return value is the estimate of the optimal parameters.
*/
RNNParametersT computeOptimalParameters(RealT R,   //半径
					RealT successProbability,     //成功率
					IntT nPoints,      //点的个数
					IntT dimension,     //维度
					PPointT *dataSet,    //数据集合
					IntT nSampleQueries,     //样本查询点的个数
					PPointT *sampleQueries,     //样本查询点的数据集
					MemVarT memoryUpperBound){   //每个半径需要使用内存的上界，多个半径的时候Qboundry才会使用到，用一个半径的时候，是使用不到的。
  ASSERT(nSampleQueries > 0);

  initializeLSHGlobal();   //设置了一个时间参数，把时间函数所花的时间求出来，得出精确操作所花费的时间

  RNNParametersT optParameters;
  optParameters.successProbability = successProbability;
  optParameters.dimension = dimension;
  optParameters.parameterR = R;
#ifdef USE_L1_DISTANCE
  optParameters.parameterR2 = R;   //L1 norm下R2 与R相等，   //R与R2的作用？
#else  
  optParameters.parameterR2 = SQR(R);   //？R2的作用是什么
#endif
  optParameters.useUfunctions = TRUE; // TODO: could optimize here:
				      // maybe sometimes, the old way
				      // was better.
  optParameters.parameterW = PARAMETER_W_DEFAULT;  //W用于计算L2 norm下的LSH函数的参数
  optParameters.parameterT = nPoints;    //查询点的个数
  optParameters.typeHT = HT_HYBRID_CHAINS;  //hashing表构建的类型
  
  // Compute the run-time parameters (timings of different parts of the algorithm).
  IntT nReps = 10; // # number of repetions //重复的数量为10个 
  RealT lshPrecomp = 0, uhashOver = 0, distComp = 0;  //LSH预处理的时间，从Hash表中取得桶的时间 ，计算两点距离的时间
  for(IntT i = 0; i < nReps; i++){          //重复nReps次，计算LSH 3个阶段的时间，然后在计算出来的平均时间。
    RealT lP, uO, dC;                    
    determineRTCoefficients(optParameters.parameterR, 
			    optParameters.successProbability, 
			    optParameters.useUfunctions, 
			    optParameters.typeHT, 
			    optParameters.dimension, 
			    nPoints, 
			    dataSet, 
			    lP, 
			    uO, 
			    dC);
    lshPrecomp += lP;   //累计重复nReps次的时间。       因为随机数产生是不同的，所以每次处理的时间也不同
    uhashOver += uO;            
    distComp += dC;            
    DPRINTF4("Coefs: lP = %0.9lf\tuO = %0.9lf\tdC = %0.9lf\n", lP, uO, dC);    
  }
  lshPrecomp /= nReps;     //平均处理的时间
  uhashOver /= nReps;     
  distComp /= nReps;      
  DPRINTF("Coefs (final): lshPrecomp = %0.9lf\n", lshPrecomp);    
  DPRINTF("Coefs (final): uhashOver = %0.9lf\n", uhashOver);        
  DPRINTF("Coefs (final): distComp = %0.9lf\n", distComp);          
  DPRINTF("Remaining memory: %lld\n", memoryUpperBound);         

  // Try all possible <k>s and choose the one for which the time
  // estimate of a query is minimal.
  IntT k;     
  RealT timeLSH, timeUH, timeCycling;    
  //IntT queryIndex = genRandomInt(0, nPoints);
  //PPointT query = dataSet[queryIndex]; // query points = a random points from the data set. //查询点取为从数据集中取的随机点
  IntT bestK = 0;
  RealT bestTime = 0;
  for(k = 2; ; k += 2){

    DPRINTF("ST. k = %d\n", k);
    IntT m = computeMForULSH(k, successProbability);    //通过k和p计算出m
    IntT L = m * (m-1) / 2;
    //DPRINTF("Available memory: %lld\n", getAvailableMemory());
    if (L * nPoints > memoryUpperBound / 12){   //??为什么除以12呢？？
      break;
    }
    timeLSH = m * k * lshPrecomp;    //m次，k维预处理时间
    timeUH = L * uhashOver;           //L个hash表
    //RealT nCollisions = estimateNCollisionsFromDSPoint(nPoints, dimension, dataSet, queryIndex, k, L, R);

    // Compute the mean number of collisions for the points from the sample query set.
    RealT nCollisions = 0;      //计算平均冲突的点的数量
    for(IntT i = 0; i < nSampleQueries; i++){
      nCollisions += estimateNDistinctCollisions(nPoints, dimension, dataSet, sampleQueries[i], TRUE, k, m, R);
    }
    nCollisions /= nSampleQueries;    //每个查询点能查询到近邻点的平均数量

    timeCycling = nCollisions * distComp;       //查询点查到所有近邻的时间？？
    DPRINTF3("ST.m=%d L=%d \n", m, L);
    DPRINTF("ST.Estimated # distinct collisions = %0.6lf\n", (double)nCollisions);
    DPRINTF("ST.TimeLSH = %0.6lf\n", timeLSH);
    DPRINTF("ST.TimeUH = %0.6lf\n", timeUH);
    DPRINTF("ST.TimeCycling = %0.6lf\n", timeCycling);
    DPRINTF("ST.Sum = %0.6lf\n", timeLSH + timeUH + timeCycling);
    if (bestK == 0 || (timeLSH + timeUH + timeCycling) < bestTime) {  //第一轮循环k=0,此时bestTime是比较大的，所以随着k的增大，得到最优时间
      bestK = k;
      bestTime = timeLSH + timeUH + timeCycling;
    }
    ASSERT(k < 100); //  otherwise, k reached 100 -- which, from
    //  experience, should never happen for reasonable
    //  data set & available memory amount.
  }


  DPRINTF("STO.Optimal k = %d\n", bestK);    
  IntT m = computeMForULSH(bestK, successProbability);   //计算L2 norm下的m
  IntT L = m * (m-1) / 2;
  timeLSH = m * bestK * lshPrecomp;
  timeUH = L * uhashOver;
  
  // Compute the mean number of collisions for the points from the sample query set.
  RealT nCollisions = 0;
  for(IntT i = 0; i < nSampleQueries; i++){
    nCollisions += estimateNDistinctCollisions(nPoints, dimension, dataSet, sampleQueries[i], TRUE, k, m, R);
  }
  nCollisions /= nSampleQueries;

  // timeCycling = estimateNCollisionsFromDSPoint(nPoints, dimension, dataSet, queryIndex, bestK, L, R) * distComp;
  timeCycling = nCollisions * distComp;
  DPRINTF("STO.TimeLSH = %0.6lf\n", timeLSH);
  DPRINTF("STO.TimeUH = %0.6lf\n", timeUH);
  DPRINTF("STO.TimeCycling = %0.6lf\n", timeCycling);
  DPRINTF("STO.Sum = %0.6lf\n", timeLSH + timeUH + timeCycling);
  
  optParameters.parameterK = bestK;
  optParameters.parameterM = m;
  optParameters.parameterL = L;

  return optParameters;
}
