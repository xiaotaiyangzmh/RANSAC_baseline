#include "plane_seg/IncrementalPlaneEstimator.hpp"

#include <numeric>

using namespace planeseg;

Eigen::Vector4f IncrementalPlaneEstimator::
getPlane(const Eigen::Vector3d& iSum,
         const Eigen::Matrix3d& iSumSquared,
         const double iCount) {
  /*
  This function is to fit a plane with sum and sumsquare of iCount points
  Why they don't use the same method as estimatefull of PlaneFitter.
  */

  Eigen::Vector3d mean = iSum/iCount;
  Eigen::Matrix3d cov = iSumSquared/iCount - mean*mean.transpose();
  Eigen::Vector4d plane;
  plane.head<3>() = cov.jacobiSvd(Eigen::ComputeFullV).matrixV().col(2);
  plane[3] = -plane.head<3>().dot(mean);
  return plane.cast<float>();
}    

IncrementalPlaneEstimator::
IncrementalPlaneEstimator() {
  reset();
}

void IncrementalPlaneEstimator::
reset() {
  mSum.setZero();
  mSumSquared.setZero();
  mPoints.clear();
  mCount = 0;
}

int IncrementalPlaneEstimator::
getNumPoints() const {
  return mPoints.size();
}

void IncrementalPlaneEstimator::
addPoint(const Eigen::Vector3f& iPoint) {
  mPoints.push_back(iPoint);
  Eigen::Vector3d p = iPoint.cast<double>();
  mSum += p;
  mSumSquared += p*p.transpose();
}

std::vector<float> IncrementalPlaneEstimator::
computeErrors(const Eigen::Vector4f& iPlane,
              const std::vector<Eigen::Vector3f>& iPoints) {
  const int n = iPoints.size();
  std::vector<float> errors(n);
  for (int i = 0; i < n; ++i) {
    errors[i] = computeError(iPlane, iPoints[i]);
  }
  return errors;
}

bool IncrementalPlaneEstimator::
tryPoint(const Eigen::Vector3f& iPoint, const float iMaxError) {
  /*
  This function is to get whether the new point iPoint can be added 
  to the current points mPoints as an inlier.

  First, get the new plane with both iPoint and mPoints.
  Then, compute the error of each point from the plane, including mPoints and iPoint.
  If all errors are smaller than the threshold (all n+1 points are inliers),
  the new point iPoint can be added to mPoints to form a new plane.
  */

  const int n = mPoints.size();
  if (n <= 2) return true;
  Eigen::Vector3d p = iPoint.cast<double>();
  Eigen::Vector3d sum = mSum + p;
  Eigen::Matrix3d sumSquared = mSumSquared + p*p.transpose();
  Eigen::Vector4f plane = getPlane(sum, sumSquared, n+1);
  std::vector<float> errors2 = computeErrors(plane, mPoints);
  errors2.push_back(computeError(plane, iPoint));
  float thresh2 = iMaxError*iMaxError;
  int numInliers = 0;
  float totalError2 = 0;
  for (float e2 : errors2) {
    totalError2 += e2;
    numInliers += (e2<=thresh2);
  }
  return numInliers==(n+1);
}

bool IncrementalPlaneEstimator::
tryPoint(const Eigen::Vector3f& iPoint, const Eigen::Vector3f& iNormal,
         const float iMaxError, const float iMaxAngle) {
  /*
  This function is to get whether the new point iPoint can be added 
  to the current points mPoints as an inlier.

  First, get the new plane with both iPoint and mPoints, and current plane with only mPoints.
  If the angle between the normal at the new point iPoint and the normal of current plane
  is larger than the angle threshold, we discard iPoint. 
  Then, compute the previous errors of mPoints from the current plane and
  new errors of mPoints and iPoint from the new plane.
  If the difference bewteen the current errors and new errors are smaller than the threshold,
  the new point iPoint can be added to mPoints to form a new plane.
  */

  const int n = mPoints.size();
  if (n < 2) return true;

  Eigen::Vector3d p = iPoint.cast<double>();
  Eigen::Vector3d sum = mSum+p;
  Eigen::Matrix3d sumSquared = mSumSquared + p*p.transpose();
  Eigen::Vector4f plane = getPlane(sum, sumSquared, n+1);
  Eigen::Vector4f currentPlane = getCurrentPlane();
  // It might be better if we compute the angle 
  // between the normal of CURRENT plane and the normal at iPoint. (?)
  // if (std::abs(plane.head<3>().cast<float>().dot(iNormal)) <
  //     std::cos(iMaxAngle)) return false;
  if (std::abs(currentPlane.head<3>().cast<float>().dot(iNormal)) <
      std::cos(iMaxAngle)) return false;

  std::vector<float> prevErrors2 = computeErrors(currentPlane, mPoints);
  // NB: Using Eigen's sum redux is significantly faster than std::accumulate
  float prevTotalError2 = Eigen::Map<Eigen::VectorXf>(prevErrors2.data(), prevErrors2.size()).sum();

  std::vector<float> errors2 = computeErrors(plane, mPoints);
  errors2.push_back(computeError(plane, iPoint));
  float thresh2 = iMaxError*iMaxError;
  int numInliers = 0;
  float totalError2 = 0;
  for (float e2 : errors2) {
    totalError2 += e2;
    numInliers += (e2<=thresh2);
  }
  float deltaError2 = totalError2/(n+1) - prevTotalError2/n;
  return deltaError2 < thresh2/n;
  //return numInliers==(n+1);
}

Eigen::Vector4f IncrementalPlaneEstimator::
getCurrentPlane() {
  return getPlane(mSum, mSumSquared, mPoints.size());
}
