;; Auto-generated. Do not edit!


(when (boundp 'yolo_ros::TriggerSelection)
  (if (not (find-package "YOLO_ROS"))
    (make-package "YOLO_ROS"))
  (shadow 'TriggerSelection (find-package "YOLO_ROS")))
(unless (find-package "YOLO_ROS::TRIGGERSELECTION")
  (make-package "YOLO_ROS::TRIGGERSELECTION"))
(unless (find-package "YOLO_ROS::TRIGGERSELECTIONREQUEST")
  (make-package "YOLO_ROS::TRIGGERSELECTIONREQUEST"))
(unless (find-package "YOLO_ROS::TRIGGERSELECTIONRESPONSE")
  (make-package "YOLO_ROS::TRIGGERSELECTIONRESPONSE"))

(in-package "ROS")



(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass yolo_ros::TriggerSelectionRequest
  :super ros::object
  :slots ())

(defmethod yolo_ros::TriggerSelectionRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass yolo_ros::TriggerSelectionResponse
  :super ros::object
  :slots (_object_names _object_positions ))

(defmethod yolo_ros::TriggerSelectionResponse
  (:init
   (&key
    ((:object_names __object_names) (let (r) (dotimes (i 0) (push "" r)) r))
    ((:object_positions __object_positions) ())
    )
   (send-super :init)
   (setq _object_names __object_names)
   (setq _object_positions __object_positions)
   self)
  (:object_names
   (&optional __object_names)
   (if __object_names (setq _object_names __object_names)) _object_names)
  (:object_positions
   (&rest __object_positions)
   (if (keywordp (car __object_positions))
       (send* _object_positions __object_positions)
     (progn
       (if __object_positions (setq _object_positions (car __object_positions)))
       _object_positions)))
  (:serialization-length
   ()
   (+
    ;; string[] _object_names
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _object_names)) 4
    ;; geometry_msgs/Point[] _object_positions
    (apply #'+ (send-all _object_positions :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string[] _object_names
     (write-long (length _object_names) s)
     (dolist (elem _object_names)
       (write-long (length elem) s) (princ elem s)
       )
     ;; geometry_msgs/Point[] _object_positions
     (write-long (length _object_positions) s)
     (dolist (elem _object_positions)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string[] _object_names
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _object_names (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _object_names i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;; geometry_msgs/Point[] _object_positions
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _object_positions (let (r) (dotimes (i n) (push (instance geometry_msgs::Point :init) r)) r))
     (dolist (elem- _object_positions)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(defclass yolo_ros::TriggerSelection
  :super ros::object
  :slots ())

(setf (get yolo_ros::TriggerSelection :md5sum-) "d609001ca88f5f9d939d2ba261852dbd")
(setf (get yolo_ros::TriggerSelection :datatype-) "yolo_ros/TriggerSelection")
(setf (get yolo_ros::TriggerSelection :request) yolo_ros::TriggerSelectionRequest)
(setf (get yolo_ros::TriggerSelection :response) yolo_ros::TriggerSelectionResponse)

(defmethod yolo_ros::TriggerSelectionRequest
  (:response () (instance yolo_ros::TriggerSelectionResponse :init)))

(setf (get yolo_ros::TriggerSelectionRequest :md5sum-) "d609001ca88f5f9d939d2ba261852dbd")
(setf (get yolo_ros::TriggerSelectionRequest :datatype-) "yolo_ros/TriggerSelectionRequest")
(setf (get yolo_ros::TriggerSelectionRequest :definition-)
      "# Request (empty)
---
# Response
string[] object_names
geometry_msgs/Point[] object_positions

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
")

(setf (get yolo_ros::TriggerSelectionResponse :md5sum-) "d609001ca88f5f9d939d2ba261852dbd")
(setf (get yolo_ros::TriggerSelectionResponse :datatype-) "yolo_ros/TriggerSelectionResponse")
(setf (get yolo_ros::TriggerSelectionResponse :definition-)
      "# Request (empty)
---
# Response
string[] object_names
geometry_msgs/Point[] object_positions

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
")



(provide :yolo_ros/TriggerSelection "d609001ca88f5f9d939d2ba261852dbd")


