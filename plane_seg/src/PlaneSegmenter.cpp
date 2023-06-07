#include "plane_seg/PlaneSegmenter.hpp"

#include <queue>
#include <pcl/search/kdtree.h>

#include "plane_seg/IncrementalPlaneEstimator.hpp"

using namespace planeseg;

PlaneSegmenter::
PlaneSegmenter() {
  setMaxError(0.02);
  setMaxAngle(30);
  setSearchRadius(0.03);
  setMinPoints(500);
}

void PlaneSegmenter::
setData(const LabeledCloud::Ptr& iCloud,
        const NormalCloud::Ptr& iNormals) {
  mCloud = iCloud;
  mNormals = iNormals;
}

void PlaneSegmenter::
setMaxError(const float iError) {
  mMaxError = iError;
}

void PlaneSegmenter::
setMaxAngle(const float iAngle) {
  mMaxAngle = iAngle*M_PI/180;
}

void PlaneSegmenter::
setSearchRadius(const float iRadius) {
  mSearchRadius = iRadius;
}


void PlaneSegmenter::
setMinPoints(const int iMin) {
  mMinPoints = iMin;
}

PlaneSegmenter::Result PlaneSegmenter::
go() {
  Result result;
  const int n = mCloud->size();

  // ------------ get nearest neighbors of each point ------------
  // create kdtree
  pcl::search::KdTree<Point>::Ptr tree
    (new pcl::search::KdTree<Point>());
  tree->setInputCloud(mCloud);

  // define the nearest neighbors and the distances from the point to its nearest neighbors
  // neighbors = {{nearest neighbors of point 1},
  //              {nearest neighbors of point 2},
  //              ...,
  //              {nearest neighbors of point n}}
  std::vector<std::vector<int>> neighbors(n);
  std::vector<float> distances;
  for (int i = 0; i < n; ++i) { 
    // get nearest neighbors of point i
    tree->radiusSearch(i, mSearchRadius, neighbors[i], distances); // new, give a max number of points (********)
    auto& neigh = neighbors[i];
    // save the nearest neighbors of point i and distance from neighbors to point i
    std::vector<std::pair<int,float>> pairs(neigh.size());
    for (int j = 0; j < (int)neigh.size(); ++j) {
      pairs[j].first = neigh[j];
      pairs[j].second = distances[j];
    }
    // sort the nearest neighbors of point i with respect to distances 
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<int,float>& iA,
                 const std::pair<int,float>& iB){
                return iA.second<iB.second;});
    // save the sorted neighbors
    for (int j = 0; j < (int)neigh.size(); ++j) {
      neigh[j] = pairs[j].first;
    }
  }

  // hitmask
  // record curvature at each point is + or -, for curvature < 0, hitMask = True
  // the output shows that no curvature is smaller than 0
  std::vector<bool> hitMask(n);
  for (int i = 0; i < n; ++i) {
    hitMask[i] = (mNormals->points[i].curvature < 0);
  }

  // create list of points ordered by curvature
  // allIndices contains the indices of points at which the curvature >= 0
  // and the curvatures are sorted in incresing order
  std::vector<int> allIndices;
  allIndices.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (!hitMask[i]) allIndices.push_back(i);
  }

  std::sort(allIndices.begin(), allIndices.end(),
            [this](const int iA, const int iB)
            { return mNormals->points[iA].curvature <
              mNormals->points[iB].curvature; });

  // create labels for each point
  std::vector<int> labels(n);
  // std::fill(labels.begin(), labels.end(), 0); std::fill(labels.begin(), labels.end(), -1); 
  std::fill(labels.begin(), labels.end(), 0); 
  IncrementalPlaneEstimator planeEst;
  // int curLabel = 1; int curLabel = 0; 
  int curLabel = 1; 
  std::deque<int> workQueue;

  // create label-to-plane mapping
  // creat a vector for each plane
  struct Plane {
    Eigen::Vector4f mPlane;
    int mCount;
    int mLabel;
  };
  std::vector<Plane> planes;

  // process the point iIndex
  // firstly, check if point iIndex (pt) can be added to the current plane
  // if yes, add pt to current plane (planeEst.mPoints)
  // then check the neighbors of point iIndex (pt), if it hasn't been added to a plane, add this point to the queue
  // the point can be added repeadly but once it has been added to a plane, it will not be processed again
  auto processPoint = [&](const int iIndex) {
    if (hitMask[iIndex]) return false;
    const Eigen::Vector3f& pt = mCloud->points[iIndex].getVector3fMap();
    const auto& cloudNorm = mNormals->points[iIndex];
    const Eigen::Vector3f norm(cloudNorm.normal_x, cloudNorm.normal_y,
                               cloudNorm.normal_z);
    if (planeEst.tryPoint(pt, norm, mMaxError, mMaxAngle)) {
      labels[iIndex] = curLabel;
      hitMask[iIndex] = true;
      planeEst.addPoint(pt);
      for (const auto idx : neighbors[iIndex]) {
        // if (!hitMask[idx] && (labels[idx]<0)) if (!hitMask[idx] && (labels[idx]<=0))
        if (!hitMask[idx] && (labels[idx]<=0)) workQueue.push_back(idx);
      }
      return true;
    }
    return false;
  };

  // iterate over points
  for (const auto idx : allIndices) {
    if (hitMask[idx]) continue;
    // if (labels[idx] >= 0) continue; if (labels[idx] > 0) continue; 
    if (labels[idx] > 0) continue; 

    // start new component
    // the estimation starts at point idx
    // for the first point (point idx), tryPoint returns true, 
    // the neighbors of the first point (point idx) are added to the queue;
    // then comes to the first neighbor, tryPoint returns true because there is only one point now in the current plane (planeEst.mPoints),
    // and the neighbors of the first neighbor are added to the queue;
    // then comes to the second neighbor and goes on ...
    // the process stops when workQueue becomes empty, which means all points have been processed,
    // no points can be added to the current plane
    planeEst.reset();
    workQueue.clear();
    workQueue.push_back(idx);

    // an example to use std::deque https://www.geeksforgeeks.org/deque-cpp-stl/
    while (workQueue.size() > 0) {
      processPoint(workQueue.front());
      workQueue.pop_front();
    }

    // add new plane
    // std::cout << planeEst.getNumPoints() << ", " << std::flush;
    Plane plane;
    plane.mPlane = planeEst.getCurrentPlane();
    plane.mCount = planeEst.getNumPoints();
    plane.mLabel = curLabel;
    planes.push_back(plane);

    ++curLabel;
  }

  // original code
  // unlabel small components
  std::sort(planes.begin(), planes.end(),
            [](const Plane& iA, const Plane& iB) {
              return iA.mCount>iB.mCount;});
  std::vector<int> lut(curLabel);
  for (int i = 0; i < (int)lut.size(); ++i) lut[i] = i;
  for (int i = 0; i < (int)planes.size(); ++i) {
    if (planes[i].mCount < mMinPoints) lut[planes[i].mLabel] = -1;
  }
  for (int i = 0; i < n; ++i) {
    if (!hitMask[i]) continue;
    int newLabel = lut[labels[i]];
    if (newLabel < 0) {
      hitMask[i] = false;
      labels[i] = 0;
    }
  }

  // remap labels
  std::unordered_map<int,Plane> planeMap;
  for (const auto& plane : planes) planeMap[plane.mLabel] = plane;
  int counter = 1;
  std::unordered_map<int,Eigen::Vector4f> planeMapNew;
  for (int& idx : lut) {
    if (idx <= 0) continue;
    Plane plane = planeMap[idx];
    idx = counter++;
    plane.mLabel = idx;
    planeMapNew[idx] = plane.mPlane;
  }
  for (int i = 0; i < n; ++i) {
    auto& label = labels[i];
    label = lut[label];
  }
    
  result.mLabels = labels;
  result.mPlanes = planeMapNew;



  // // unlabel small components, including hitMask and labels of points on these small planes 
  // // for valid planes lut is their labels, for small planes lut is -1
  // std::vector<int> lut(curLabel);
  // std::fill(lut.begin(), lut.end(), -1);
  // for (int i = 0; i < (int)planes.size(); ++i) {
  //   if (planes[i].mCount >= mMinPoints) lut[planes[i].mLabel] = planes[i].mLabel;
  //   else planes[i].mLabel = -1;
  // }
  // for (int i = 0; i < n; ++i) {
  //   if (!hitMask[i]) continue;
  //   // labels[i] is the plane to which the point i belongs
  //   // newLabel is either -1 when the plane is too small, or the label of the plane
  //   int newLabel = lut[labels[i]];
  //   if (newLabel < 0) {
  //     hitMask[i] = false;
  //     labels[i] = -1; // labels[i] = 0;
  //   }
  // }

  // // remap labels
  // std::unordered_map<int,Eigen::Vector4f> planeMap;
  // for (const auto& plane : planes) {
  //   if (plane.mLabel < 0) continue;
  //   planeMap[plane.mLabel] = plane.mPlane;
  // }

  // result.mLabels = labels;
  // result.mPlanes = planeMap;

  return result;
}
