#pragma once

// Local navigation for a model-selected FINAL viewpoint. No person heading,
// scene model, task words or model API is used here. Unknown cells are blocked.
#include <vla_sim/planner_trajectory_safety.h>
#include <queue>
#include <limits>

namespace vla_sim {

template<class Free>
bool routeLineFree(const Eigen::Vector3d &a, const Eigen::Vector3d &b, Free free) {
  const int n=std::max(1,static_cast<int>(std::ceil((b-a).norm()/.06)));
  for(int i=0;i<=n;++i) if(!free(a+(b-a)*(static_cast<double>(i)/n))) return false;
  return true;
}

// Bounded 2-D A*: the camera stays at the authorized altitude. Returns the
// exact goal if reachable, otherwise the best observed frontier. A frontier is
// an internal sensor horizon; it does NOT replace the model's final goal.
template<class Free,class Reference>
std::vector<Eigen::Vector3d> localPhotoRoute(const Eigen::Vector3d &start,
    const Eigen::Vector3d &goal, const Eigen::Vector4d &bounds, Free free, Reference reference) {
  if(!start.allFinite() || !goal.allFinite() || std::abs(start.z()-goal.z())>.05 || !free(start)) return {};
  if(routeLineFree(start,goal,free)) return {start,goal};
  constexpr double step=.25;
  const int xmin=static_cast<int>(std::ceil((bounds[0]-start.x())/step));
  const int xmax=static_cast<int>(std::floor((bounds[1]-start.x())/step));
  const int ymin=static_cast<int>(std::ceil((bounds[2]-start.y())/step));
  const int ymax=static_cast<int>(std::floor((bounds[3]-start.y())/step));
  const int nx=xmax-xmin+1,ny=ymax-ymin+1;
  if(nx<=0 || ny<=0 || nx*ny>30000 || xmin>0 || xmax<0 || ymin>0 || ymax<0) return {};
  const auto index=[&](int x,int y){return (x-xmin)*ny+y-ymin;};
  const auto point=[&](int id)->Eigen::Vector3d {return start+Eigen::Vector3d((id/ny+xmin)*step,(id%ny+ymin)*step,0.);};
  const int first=index(0,0),total=nx*ny;
  // A reference cost-to-go prevents a greedy frontier from sticking to the
  // front of a person/obstacle. It authorizes NO unknown-space flight.
  std::vector<double> remaining(total,std::numeric_limits<double>::infinity());
  using Item=std::pair<double,int>;
  std::priority_queue<Item,std::vector<Item>,std::greater<Item>> guide;
  const int gx=std::max(xmin,std::min(xmax,static_cast<int>(std::round((goal.x()-start.x())/step))));
  const int gy=std::max(ymin,std::min(ymax,static_cast<int>(std::round((goal.y()-start.y())/step))));
  const int final_index=index(gx,gy);
  if(reference(point(final_index))) {remaining[final_index]=0.;guide.emplace(0.,final_index);}
  std::vector<int> reference_state(total,-1);
  while(!guide.empty()) {
    const auto top=guide.top();guide.pop();
    const int at=top.second;
    if(top.first>remaining[at]+1e-9)continue;
    const int x=at/ny+xmin,y=at%ny+ymin;
    for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy) {
      if((!dx&&!dy)||x+dx<xmin||x+dx>xmax||y+dy<ymin||y+dy>ymax)continue;
      const int next=index(x+dx,y+dy);
      if(reference_state[next]<0)reference_state[next]=reference(point(next))?1:0;
      if(!reference_state[next] || !reference((point(at)+point(next))*.5))continue;
      const double distance=top.first+step*std::hypot(dx,dy);
      if(distance+1e-9<remaining[next]) {remaining[next]=distance;guide.emplace(distance,next);}
    }
  }
  std::vector<double> cost(total,std::numeric_limits<double>::infinity());
  std::vector<int> parent(total,-1),state(total,-1);
  std::vector<bool> closed(total,false);
  std::priority_queue<Item,std::vector<Item>,std::greater<Item>> open;
  const auto heuristic=[&](int i){return std::isfinite(remaining[i])?remaining[i]:(point(i)-goal).norm()+100.;};
  cost[first]=0.;open.emplace(heuristic(first),first);
  int best=first,terminal=-1,expanded=0;
  double best_h=heuristic(first);
  while(!open.empty() && expanded++<30000) {
    const int current=open.top().second;open.pop();
    if(closed[current]) continue;
    closed[current]=true;
    const Eigen::Vector3d p=point(current);
    const double h=(p-goal).norm();
    if(heuristic(current)<best_h) {best_h=heuristic(current);best=current;}
    if(h<step*1.5 && routeLineFree(p,goal,free)) {terminal=current;break;}
    const int x=current/ny+xmin,y=current%ny+ymin;
    for(int dx=-1;dx<=1;++dx) for(int dy=-1;dy<=1;++dy) {
      if((dx==0&&dy==0)||x+dx<xmin||x+dx>xmax||y+dy<ymin||y+dy>ymax) continue;
      const int next=index(x+dx,y+dy);
      if(closed[next]) continue;
      const Eigen::Vector3d q=point(next);
      if(state[next]<0) state[next]=free(q)?1:0;
      if(!state[next] || !free((p+q)*.5)) continue;
      const double next_cost=cost[current]+(q-p).norm();
      if(next_cost+1e-9>=cost[next]) continue;
      cost[next]=next_cost;parent[next]=current;
      open.emplace(next_cost+heuristic(next),next);
    }
  }
  const int last=terminal<0?best:terminal;
  if(last==first && terminal<0) return {};
  std::vector<Eigen::Vector3d> reverse;
  for(int at=last;at>=0;at=parent[at]) {reverse.push_back(point(at));if(at==first)break;}
  if(reverse.empty() || (reverse.back()-start).norm()>1e-6) return {};
  std::reverse(reverse.begin(),reverse.end());
  if(terminal>=0 && (reverse.back()-goal).norm()>1e-6) reverse.push_back(goal);
  // Remove grid staircases with measured line-of-sight checks. No chord can
  // cut a human zone or an unseen obstacle corner.
  std::vector<Eigen::Vector3d> result{start};
  for(size_t i=0;i+1<reverse.size();) {
    size_t next=i+1;
    for(size_t j=i+2;j<reverse.size();++j) if(routeLineFree(reverse[i],reverse[j],free)) next=j;
    result.push_back(reverse[next]);i=next;
  }
  if((result.back()-start).norm()<.20 && terminal<0) return {};
  return result;
}

template<class Free>
std::vector<Eigen::Vector3d> localPhotoRoute(const Eigen::Vector3d &start,
    const Eigen::Vector3d &goal,const Eigen::Vector4d &bounds,Free free) {
  return localPhotoRoute(start,goal,bounds,free,free);
}

inline std::vector<Eigen::Vector3d> routePrefix(const std::vector<Eigen::Vector3d> &path,double horizon) {
  if(path.empty()) return {};
  std::vector<Eigen::Vector3d> result{path.front()};
  for(size_t i=1;i<path.size();++i) {
    const double d=(path[i]-result.back()).norm();
    if(d>horizon) {result.push_back(result.back()+(path[i]-result.back())*(horizon/d));break;}
    result.push_back(path[i]);horizon-=d;
    if(horizon<1e-6)break;
  }
  return result;
}

// MINCO fits ONE continuous curve through internal route points. Internal
// joins have continuous velocity/acceleration and are never hover goals.
template<class Free>
bool makePhotoRouteTrajectory(const std::vector<Eigen::Vector3d> &route,
    const Eigen::Vector3d &velocity,const Eigen::Vector3d &acceleration,
    double maxv,double maxa,double maxj,Free free,poly_traj::MinJerkOpt &result) {
  if(route.size()<2 || !velocity.allFinite() || !acceleration.allFinite()) return false;
  for(double spacing:{.6,.3,.15}) {
    std::vector<Eigen::Vector3d> points{route.front()};
    for(size_t i=1;i<route.size();++i) {
      const auto a=points.back();const double d=(route[i]-a).norm();
      const int n=std::max(1,static_cast<int>(std::ceil(d/spacing)));
      for(int k=1;k<=n;++k) points.push_back(a+(route[i]-a)*(static_cast<double>(k)/n));
    }
    if(points.size()<3)points.insert(points.begin()+1,(points.front()+points.back())*.5);
    const int n=static_cast<int>(points.size())-1;
    if(n>96) return false;
    Eigen::Matrix3d head,tail;
    head<<points.front(),velocity,acceleration;
    tail<<points.back(),Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero();
    Eigen::MatrixXd inner(3,n-1);Eigen::VectorXd times(n);
    for(int i=0;i<n;++i) {
      times[i]=std::max(.5,(points[i+1]-points[i]).norm()/(maxv*.65));
      if(i+1<n)inner.col(i)=points[i+1];
    }
    for(int attempt=0;attempt<7;++attempt) {
      result.reset(head,tail,n);result.generate(inner,times);
      const auto trajectory=result.getTraj();
      if(!finiteTrajectory(trajectory) || trajectory.getTotalDuration()>90.) break;
      double jerk=0.;
      for(const auto &piece:trajectory) {
        const auto c=piece.getCoeffMat();const double t=piece.getDuration();
        const Eigen::Vector3d j0=6.*c.col(2),j1=j0+12.*t*c.col(1),j2=j0+24.*t*c.col(1)+60.*t*t*c.col(0);
        jerk=std::max({jerk,j0.norm(),j1.norm(),j2.norm()});
      }
      const double scale=std::max({trajectory.getMaxVelRate()/maxv,
          std::sqrt(trajectory.getMaxAccRate()/maxa),std::cbrt(jerk/maxj)});
      if(scale>1.+1e-6) {times*=std::max(1.1,scale*1.05);continue;}
      if(shortRecoveryKnownFree(trajectory,free)) return true;
      break; // Geometric overshoot needs denser route support, not slower time.
    }
  }
  return false;
}
} // namespace vla_sim
