#pragma once
#include <Eigen/Core>
#include <cmath>

// Every voxel intersected by the finite cylindrical flight envelope must have
// explicit free evidence. Checking only the center permits a one-ray corridor.
// The predicate receives global voxel indices and must reject outside-map cells.
template<class KnownFree>
bool cylinderKnownFree(const Eigen::Vector3d &p, double resolution,
                       double radius, double half_height, KnownFree known) {
  const Eigen::Vector3d extent(radius,radius,half_height);
  const Eigen::Vector3i lo=((p-extent)/resolution).array().floor().cast<int>();
  const Eigen::Vector3i hi=((p+extent)/resolution).array().floor().cast<int>();
  for(int x=lo.x();x<=hi.x();++x) for(int y=lo.y();y<=hi.y();++y) {
    // Closest XY point of each voxel's AABB to the cylinder axis.
    const double dx=std::max(0.0,std::max(x*resolution-p.x(),p.x()-(x+1)*resolution));
    const double dy=std::max(0.0,std::max(y*resolution-p.y(),p.y()-(y+1)*resolution));
    if(dx*dx+dy*dy>radius*radius) continue;
    for(int z=lo.z();z<=hi.z();++z)
      if(!known(Eigen::Vector3i(x,y,z))) return false;
  }
  return true;
}
