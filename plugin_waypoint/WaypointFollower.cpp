#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <fstream>
#include <vector>

namespace gazebo {
class WaypointFollower : public ModelPlugin {
public:
  physics::ModelPtr model;
  std::vector<ignition::math::Vector3d> waypoints;
  int index = 0;
  double speed = 1.0;

  void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) {
      model = _model;

      std::string file = _sdf->Get<std::string>("waypoints_file");
      std::ifstream in(file);

      double x, y, z;
      char comma;
      while (in >> x >> comma >> y >> comma >> z)
          waypoints.emplace_back(x, y, z);

      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&WaypointFollower::OnUpdate, this));
  }

  void OnUpdate() {
      if (index >= waypoints.size()) return;
      auto target = waypoints[index];
      auto pos = model->WorldPose().Pos();

      ignition::math::Vector3d dir = target - pos;
      if (dir.Length() < 0.2) { index++; return; }

      dir.Normalize();
      ignition::math::Vector3d vel = dir * speed;
      model->SetLinearVel(vel);
  }

private:
  event::ConnectionPtr updateConnection;
};

GZ_REGISTER_MODEL_PLUGIN(WaypointFollower)
}