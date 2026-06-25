#pragma once

#include <yarp/os/LogComponent.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/Node.h>
#include <yarp/os/Publisher.h>
#include <yarp/os/Time.h>
// #include <yarp/rosmsg/std_msgs/Int64.h>
// #include <yarp/rosmsg/geometry_msgs/Pose.h>
#include <yarp/rosmsg/SharedData.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>

using yarp::os::Network;
using yarp::os::Node;
using yarp::os::Publisher;

namespace {
    YARP_LOG_COMPONENT(TALKER, "yarp.example.ros.talker")
    constexpr double loop_delay = 0.1;
}

class YarpToRos
{

private:
    int udp_sock_ = -1;
    struct sockaddr_in udp_addr_;
public:

    // yarp::os::Publisher<yarp::rosmsg::geometry_msgs::Pose> publisher;
    yarp::os::Publisher<yarp::rosmsg::SharedData> port;  // changed Port to Publisher

    yarp::os::Node* node = nullptr;

    // yarp::rosmsg::geometry_msgs::Pose data;

    yarp::rosmsg::SharedData d;

void initPublisher(){
        // UDP socket -> ROS2 receiver (twist_bridge_receiver.py listening on 9870)
        udp_sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        if(udp_sock_ < 0) {
            yCError(TALKER) << "Failed to create UDP socket";
            return;
        }
        memset(&udp_addr_, 0, sizeof(udp_addr_));
        udp_addr_.sin_family = AF_INET;
        udp_addr_.sin_port = htons(9870);
        // NOTE: target IP. See the cross-container note below.
        inet_pton(AF_INET, "127.0.0.1", &udp_addr_.sin_addr);
        yInfo() << "UDP pose sender initialised -> 127.0.0.1:9870";
    }

    void publishTargetPos(double x, double y, double z, double qx, double qy, double qz, double qw){
        if(udp_sock_ < 0) return;
        // bridge expects 6 doubles: [tx, ty, tz, qx, qy, qz] in METERS.
        // qw is reconstructed on the receiver side as sqrt(1 - qx^2-qy^2-qz^2),
        // so the quaternion MUST be normalized before sending.
        double buf[6] = { x, y, z, qx, qy, qz };
        sendto(udp_sock_, buf, sizeof(buf), 0,
               (struct sockaddr*)&udp_addr_, sizeof(udp_addr_));
    }

~YarpToRos()
    {
        if(udp_sock_ >= 0) { close(udp_sock_); udp_sock_ = -1; }
        if (node) { delete node; node = nullptr; }
    }

};

