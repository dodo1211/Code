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

#ifndef GEOMETRY_INCLUDED
#define GEOMETRY_INCLUDED

// A simple point in d-dimensional space. A point is defined by a
// vector of coordinates. 
typedef struct _PointT {
  //IntT dimension;
  IntT index; // the index of this point in the dataset list of points
  RealT *coordinates;
  RealT sqrLength; // the square of the length of the vector  //向量的长度
} PointT, *PPointT;

// Encapsulates a PPoint and a RealT in a single struct.
typedef struct _PPointAndRealTStructT {
  PPointT ppoint;
  RealT real;        //存放数据点和查询点的距离
} PPointAndRealTStructT;

int comparePPointAndRealTStructT(const void *a, const void *b);

RealT distance(IntT dimension, PPointT p1, PPointT p2);

#endif
