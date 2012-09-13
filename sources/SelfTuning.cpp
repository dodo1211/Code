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
//���㺯�����е�ʱ�䣬������Ҫ��ȥ���ֵ���õ���ȷ�Ĵ���ʱ��
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


  PPointAndRealTStructT *distToNN = NULL;  //��Ų�ѯ�㣬(ppoint,real)  ppoint=(index,coordinates,sqrLength)
  FAILIF(NULL == (distToNN = (PPointAndRealTStructT*)MALLOC(nQueries * sizeof(*distToNN))));
  for(IntT i = 0; i < nQueries; i++){      //������С���룬����ÿ����ѯ���������룬û�а����ݼ��ľ�������ȥ ��ֻ�����˾��룬û�д�����������ݵ�
    distToNN[i].ppoint = queries[i];
    distToNN[i].real = distance(dimension, queries[i], dataSet[0]);
    for(IntT p = 0; p < nPoints; p++){      //��ѯ����һ���������ݼ���ÿ������бȽ�
      RealT dist = distance(dimension, queries[i], dataSet[p]);    //��ò�ѯ�������ݼ���ľ���
      if (dist < distToNN[i].real){               //����ѯ�������ݼ������С�������distToNN[i].real  //ע�⣬ֻ�洢�˾���
	distToNN[i].real = dist;
      }
    }
  }

  qsort(distToNN, nQueries, sizeof(*distToNN), comparePPointAndRealTStructT);   //�����鰴�ղ�ѯ������Ӧ������ڵľ�����������Ϊ�˶��뾶��

  IntT radiusIndex = 0;
  for(IntT i = 0; i < nQueries; i++) {
    //DPRINTF("%0.6lf\n", distToNN[i].real);
    queries[i] = distToNN[i].ppoint; // copy the sorted queries array back to <queries>   //��������֮������鵽queries
    while ((distToNN[i].real > radii[radiusIndex]) && (radiusIndex < nRadii)) {           //ÿ���뾶ֻ��¼һ���߽�ֵ
      boundaryIndeces[radiusIndex] = i;                                              //����һ�����ڰ뾶�ĵ��������Ŵ���boundary
      radiusIndex++;
    }
  }

  FREE(distToNN);
}

// Determines the run-time coefficients of the different parts of the  //ȷ����ѯ�㷨��ͬ���ֵ�����ʱ��
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

  // use a subset of the original data set.   ʹ��ԭʼ���ݼ���һ���Ӽ�
  // there is not much theory behind the formula below.    //��С�����ģ
  IntT n = nPoints / 50;    //�������n���㣬��С50��
  if (n < 100) {            //������ɵĵ�ĸ���С��100����ʹͰ�����������ݼ��������һ����
    n = nPoints;
  }
  if (n > 10000) {
    n = 10000;
  }

  // Initialize the data set to use.
  PPointT *dataSet;
  FAILIF(NULL == (dataSet = (PPointT*)MALLOC(n * sizeof(PPointT))));
  for(IntT i = 0; i < n; i++){           //����ʵ���ݼ������ȡn���� �����10000����
    dataSet[i] = realData[genRandomInt(0, nPoints - 1)];
  }

  IntT hashTableSize = n;                //��ϣ���СҲ��ʼ��Ϊn
  RNNParametersT algParameters;
  algParameters.parameterR = thresholdR;   //�뾶
  algParameters.successProbability = successProbability;
  algParameters.dimension = dimension;
#ifdef USE_L1_DISTANCE
  algParameters.parameterR2 = thresholdR;       //ʹ��L1���룬R2=R
#else  
  algParameters.parameterR2 = SQR(thresholdR);   //ʹ��L2  R2=R^2
#endif
  algParameters.useUfunctions = useUfunctions;
  algParameters.parameterK = 16;       //k �趨Ϊ16��ֻ�ǲ��ԣ���������ʱ�䣬���������������һ��ʱ�䣬֮�����ڴ����иĳ�16����Ϊ16��bestK.
  algParameters.parameterW = PARAMETER_W_DEFAULT;    //W=4��manuel��˵������β��ԣ�4����õ�ֵ
  algParameters.parameterT = n;                     //��ĸ���
  algParameters.typeHT = typeHT;                      //Ͱ������

  if (algParameters.useUfunctions){
    algParameters.parameterM = computeMForULSH(algParameters.parameterK, algParameters.successProbability);     //�����Ľ���L��M
    algParameters.parameterL = algParameters.parameterM * (algParameters.parameterM - 1) / 2;
  }else{
    algParameters.parameterM = computeLfromKP(algParameters.parameterK, algParameters.successProbability);          //���������M=L 
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
  BooleanT tempTimingOn = timingOn;    //��ʼ��ΪTrue
  timingOn = TRUE;

  // initialize result arrays
  PPointT *result = NULL;             //������Լ����ʼ��
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
      nnStruct = initLSH_WithDataSet(algParameters, n, dataSet);   //��ʼ�����ݽṹ������������ĸ��������ݼ����Ե����ӳ��ת����Ͱ����ӳ��ת���������Ͱ��
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
      queryPoint = realData[genRandomInt(0, nPoints - 1)];   //��ѯ��Ϊ���ݼ��������ȡ������һ����
      timeComputeULSH = 0;
      timeGetBucket = 0;
      timeCycleBucket = 0;
      nOfDistComps = 0;                //�����ȽϵĴ���
      nNNs = getNearNeighborsFromPRNearNeighborStruct(nnStruct, queryPoint, result, resultSize);   //����һ���������ڼ���Ƿ���������
      //DPRINTF("Time to compute LSH: %0.6lf\n", timeComputeULSH);
      //DPRINTF("Time to get bucket: %0.6lf\n", timeGetBucket);
      //DPRINTF("Time to cycle through buckets: %0.9lf\n", timeCycleBucket);
      //DPRINTF("N of dist comp: %d\n", nOfDistComps);

      ASSERT(nNNs == 0);
      if (nOfDistComps >= MIN(n / 10, 100)){    //���㹻�ĵ�ȽϹ����Ž�ʱ�����
	nSucReps++;
	lshPrecomp += timeComputeULSH / algParameters.parameterK / algParameters.parameterM;  //ÿ�����һ��ά�ȴ����hashingͰ��ʱ��
	uhashOver += timeGetBucket / algParameters.parameterL;     //ȡ��Ͱ��ʱ��
	distComp += timeCycleBucket / nOfDistComps;
      }
    }

    if (nSucReps >= 5){
      lshPrecomp /= nSucReps;
      uhashOver /= nSucReps;
      distComp /= nSucReps;
      DPRINTF1("RT coeffs computed.\n");
    }else{
      algParameters.parameterR *= 2; // double the radius and repeat  //�Ƚϵĵ������������뾶�����ظ��Ƚ�
      DPRINTF1("Could not determine the RT coeffs. Repeating.\n");
    }

    freePRNearNeighborStruct(nnStruct);

  }while(nSucReps < 5);       //��һ����Чֵ���жϣ�Ҫ���5����Чֵ

  FREE(dataSet);
  FREE(result);

  timingOn = tempTimingOn;
}

/*
  The function <p> from the paper (probability of collision of 2
  points for 1 LSH function).
 */
RealT computeFunctionP(RealT w, RealT c){   //L2 norm  //��������һ��Ͱ�г�ͻ�Ŀ����� p1
  RealT x = w / c;
  return 1 - ERFC(x / M_SQRT2) - M_2_SQRTPI / M_SQRT2 / x * (1 - EXP(-SQR(x) / 2));  //ERFCΪ��������M_SQRT2Ϊmath.h�еĺ�����ֵΪpi
														//M_2_SQRTPI= 2/sqrt(pi) ,exp(x)=e^x ����ʽ1
}

// Computes the parameter <L> of the algorithm, given the parameter
// <k> and the desired success probability
// <successProbability>. Functions <g> are considered all independent
// (original scheme).
IntT computeLfromKP(IntT k, RealT successProbability){
  return CEIL(LOG(1 - successProbability) / LOG(1 - POW(computeFunctionP(PARAMETER_W_DEFAULT, 1), k)));   //����ȡ��,LOG=ln ����ʽ6������L��ֵ
}

// Computes the parameter <m> of the algorithm, given the parameter
// <k> and the desired success probability <successProbability>. Only
// meaningful when functions <g> are interdependent (pairs of
// functions <u>, where the <m> functions <u> are each k/2-tuples of
// independent LSH functions).    //g���������ʱ�����ã� Ϊ�˼��ټ���ʱ��
IntT computeMForULSH(IntT k, RealT successProbability){
  ASSERT((k & 1) == 0); // k should be even in order to use ULSH.
  RealT mu = 1 - POW(computeFunctionP(PARAMETER_W_DEFAULT, 1), k / 2);    //1-p1^(k/2)   ��Ϊ1 ˵��c=1
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
      RealT dist = distance(dim, query, dataSet[i]);     //������Ͱ�в�ѯ�������ݼ��е�ľ���
      sumCollisions += POW(computeFunctionP(PARAMETER_W_DEFAULT, dist / R), k);    // dist���п��ܾ��Ǳ�ʾr2��distԽ��������ĸ���ԽС��
    }
  }
  return L * sumCollisions;        //���ͻ������ֵ   //sumCollosions��������Ͱ�еĳ�ͻ�����Ŀ
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
	sumCollisions += 1-POW(1-POW(computeFunctionP(PARAMETER_W_DEFAULT, dist / R), k), LorM);        //1-��1-p1^k��^L�������ͻ�ĸ���
      }else{
	RealT mu = 1 - POW(computeFunctionP(PARAMETER_W_DEFAULT, dist / R), k / 2);
	RealT x = POW(mu, LorM - 1);
	sumCollisions += 1 - mu * x - LorM * (1 - mu) * x;    //��ѯ�������ݼ����ͻ�ĸ����ܺͣ�Ϊ�ܲ�ѯ���Ľ��ڵ����Ŀ
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
//�����ѯ�������������ݼ�����Ϊ�˿�����ײ���ƣ�������Ӧ��������ѯ�㿼�ǽ����ݼ���
  The return value is the estimate of the optimal parameters.
*/
RNNParametersT computeOptimalParameters(RealT R,   //�뾶
					RealT successProbability,     //�ɹ���
					IntT nPoints,      //��ĸ���
					IntT dimension,     //ά��
					PPointT *dataSet,    //���ݼ���
					IntT nSampleQueries,     //������ѯ��ĸ���
					PPointT *sampleQueries,     //������ѯ������ݼ�
					MemVarT memoryUpperBound){   //ÿ���뾶��Ҫʹ���ڴ���Ͻ磬����뾶��ʱ��Qboundry�Ż�ʹ�õ�����һ���뾶��ʱ����ʹ�ò����ġ�
  ASSERT(nSampleQueries > 0);

  initializeLSHGlobal();   //������һ��ʱ���������ʱ�亯��������ʱ����������ó���ȷ���������ѵ�ʱ��

  RNNParametersT optParameters;
  optParameters.successProbability = successProbability;
  optParameters.dimension = dimension;
  optParameters.parameterR = R;
#ifdef USE_L1_DISTANCE
  optParameters.parameterR2 = R;   //L1 norm��R2 ��R��ȣ�   //R��R2�����ã�
#else  
  optParameters.parameterR2 = SQR(R);   //��R2��������ʲô
#endif
  optParameters.useUfunctions = TRUE; // TODO: could optimize here:
				      // maybe sometimes, the old way
				      // was better.
  optParameters.parameterW = PARAMETER_W_DEFAULT;  //W���ڼ���L2 norm�µ�LSH�����Ĳ���
  optParameters.parameterT = nPoints;    //��ѯ��ĸ���
  optParameters.typeHT = HT_HYBRID_CHAINS;  //hashing����������
  
  // Compute the run-time parameters (timings of different parts of the algorithm).
  IntT nReps = 10; // # number of repetions //�ظ�������Ϊ10�� 
  RealT lshPrecomp = 0, uhashOver = 0, distComp = 0;  //LSHԤ�����ʱ�䣬��Hash����ȡ��Ͱ��ʱ�� ��������������ʱ��
  for(IntT i = 0; i < nReps; i++){          //�ظ�nReps�Σ�����LSH 3���׶ε�ʱ�䣬Ȼ���ڼ��������ƽ��ʱ�䡣
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
    lshPrecomp += lP;   //�ۼ��ظ�nReps�ε�ʱ�䡣       ��Ϊ����������ǲ�ͬ�ģ�����ÿ�δ����ʱ��Ҳ��ͬ
    uhashOver += uO;            
    distComp += dC;            
    DPRINTF4("Coefs: lP = %0.9lf\tuO = %0.9lf\tdC = %0.9lf\n", lP, uO, dC);    
  }
  lshPrecomp /= nReps;     //ƽ�������ʱ��
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
  //PPointT query = dataSet[queryIndex]; // query points = a random points from the data set. //��ѯ��ȡΪ�����ݼ���ȡ�������
  IntT bestK = 0;
  RealT bestTime = 0;
  for(k = 2; ; k += 2){

    DPRINTF("ST. k = %d\n", k);
    IntT m = computeMForULSH(k, successProbability);    //ͨ��k��p�����m
    IntT L = m * (m-1) / 2;
    //DPRINTF("Available memory: %lld\n", getAvailableMemory());
    if (L * nPoints > memoryUpperBound / 12){   //??Ϊʲô����12�أ���
      break;
    }
    timeLSH = m * k * lshPrecomp;    //m�Σ�kάԤ����ʱ��
    timeUH = L * uhashOver;           //L��hash��
    //RealT nCollisions = estimateNCollisionsFromDSPoint(nPoints, dimension, dataSet, queryIndex, k, L, R);

    // Compute the mean number of collisions for the points from the sample query set.
    RealT nCollisions = 0;      //����ƽ����ͻ�ĵ������
    for(IntT i = 0; i < nSampleQueries; i++){
      nCollisions += estimateNDistinctCollisions(nPoints, dimension, dataSet, sampleQueries[i], TRUE, k, m, R);
    }
    nCollisions /= nSampleQueries;    //ÿ����ѯ���ܲ�ѯ�����ڵ��ƽ������

    timeCycling = nCollisions * distComp;       //��ѯ��鵽���н��ڵ�ʱ�䣿��
    DPRINTF3("ST.m=%d L=%d \n", m, L);
    DPRINTF("ST.Estimated # distinct collisions = %0.6lf\n", (double)nCollisions);
    DPRINTF("ST.TimeLSH = %0.6lf\n", timeLSH);
    DPRINTF("ST.TimeUH = %0.6lf\n", timeUH);
    DPRINTF("ST.TimeCycling = %0.6lf\n", timeCycling);
    DPRINTF("ST.Sum = %0.6lf\n", timeLSH + timeUH + timeCycling);
    if (bestK == 0 || (timeLSH + timeUH + timeCycling) < bestTime) {  //��һ��ѭ��k=0,��ʱbestTime�ǱȽϴ�ģ���������k�����󣬵õ�����ʱ��
      bestK = k;
      bestTime = timeLSH + timeUH + timeCycling;
    }
    ASSERT(k < 100); //  otherwise, k reached 100 -- which, from
    //  experience, should never happen for reasonable
    //  data set & available memory amount.
  }


  DPRINTF("STO.Optimal k = %d\n", bestK);    
  IntT m = computeMForULSH(bestK, successProbability);   //����L2 norm�µ�m
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
