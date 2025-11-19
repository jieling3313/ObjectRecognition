# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "yolo_ros: 0 messages, 1 services")

set(MSG_I_FLAGS "-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(yolo_ros_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/qrobot/yolo/client/src/yolo_ros/srv/TriggerSelection.srv" NAME_WE)
add_custom_target(_yolo_ros_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "yolo_ros" "/home/qrobot/yolo/client/src/yolo_ros/srv/TriggerSelection.srv" "geometry_msgs/Point"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(yolo_ros
  "/home/qrobot/yolo/client/src/yolo_ros/srv/TriggerSelection.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/yolo_ros
)

### Generating Module File
_generate_module_cpp(yolo_ros
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/yolo_ros
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(yolo_ros_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(yolo_ros_generate_messages yolo_ros_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/qrobot/yolo/client/src/yolo_ros/srv/TriggerSelection.srv" NAME_WE)
add_dependencies(yolo_ros_generate_messages_cpp _yolo_ros_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(yolo_ros_gencpp)
add_dependencies(yolo_ros_gencpp yolo_ros_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS yolo_ros_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(yolo_ros
  "/home/qrobot/yolo/client/src/yolo_ros/srv/TriggerSelection.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/yolo_ros
)

### Generating Module File
_generate_module_eus(yolo_ros
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/yolo_ros
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(yolo_ros_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(yolo_ros_generate_messages yolo_ros_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/qrobot/yolo/client/src/yolo_ros/srv/TriggerSelection.srv" NAME_WE)
add_dependencies(yolo_ros_generate_messages_eus _yolo_ros_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(yolo_ros_geneus)
add_dependencies(yolo_ros_geneus yolo_ros_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS yolo_ros_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(yolo_ros
  "/home/qrobot/yolo/client/src/yolo_ros/srv/TriggerSelection.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/yolo_ros
)

### Generating Module File
_generate_module_lisp(yolo_ros
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/yolo_ros
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(yolo_ros_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(yolo_ros_generate_messages yolo_ros_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/qrobot/yolo/client/src/yolo_ros/srv/TriggerSelection.srv" NAME_WE)
add_dependencies(yolo_ros_generate_messages_lisp _yolo_ros_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(yolo_ros_genlisp)
add_dependencies(yolo_ros_genlisp yolo_ros_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS yolo_ros_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(yolo_ros
  "/home/qrobot/yolo/client/src/yolo_ros/srv/TriggerSelection.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/yolo_ros
)

### Generating Module File
_generate_module_nodejs(yolo_ros
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/yolo_ros
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(yolo_ros_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(yolo_ros_generate_messages yolo_ros_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/qrobot/yolo/client/src/yolo_ros/srv/TriggerSelection.srv" NAME_WE)
add_dependencies(yolo_ros_generate_messages_nodejs _yolo_ros_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(yolo_ros_gennodejs)
add_dependencies(yolo_ros_gennodejs yolo_ros_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS yolo_ros_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(yolo_ros
  "/home/qrobot/yolo/client/src/yolo_ros/srv/TriggerSelection.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/yolo_ros
)

### Generating Module File
_generate_module_py(yolo_ros
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/yolo_ros
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(yolo_ros_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(yolo_ros_generate_messages yolo_ros_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/qrobot/yolo/client/src/yolo_ros/srv/TriggerSelection.srv" NAME_WE)
add_dependencies(yolo_ros_generate_messages_py _yolo_ros_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(yolo_ros_genpy)
add_dependencies(yolo_ros_genpy yolo_ros_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS yolo_ros_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/yolo_ros)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/yolo_ros
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(yolo_ros_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/yolo_ros)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/yolo_ros
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(yolo_ros_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/yolo_ros)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/yolo_ros
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(yolo_ros_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/yolo_ros)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/yolo_ros
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(yolo_ros_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/yolo_ros)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/yolo_ros\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/yolo_ros
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(yolo_ros_generate_messages_py geometry_msgs_generate_messages_py)
endif()
