
(cl:in-package :asdf)

(defsystem "yolo_ros-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
)
  :components ((:file "_package")
    (:file "TriggerSelection" :depends-on ("_package_TriggerSelection"))
    (:file "_package_TriggerSelection" :depends-on ("_package"))
  ))