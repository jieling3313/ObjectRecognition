; Auto-generated. Do not edit!


(cl:in-package yolo_ros-srv)


;//! \htmlinclude TriggerSelection-request.msg.html

(cl:defclass <TriggerSelection-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass TriggerSelection-request (<TriggerSelection-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TriggerSelection-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TriggerSelection-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name yolo_ros-srv:<TriggerSelection-request> is deprecated: use yolo_ros-srv:TriggerSelection-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TriggerSelection-request>) ostream)
  "Serializes a message object of type '<TriggerSelection-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TriggerSelection-request>) istream)
  "Deserializes a message object of type '<TriggerSelection-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TriggerSelection-request>)))
  "Returns string type for a service object of type '<TriggerSelection-request>"
  "yolo_ros/TriggerSelectionRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TriggerSelection-request)))
  "Returns string type for a service object of type 'TriggerSelection-request"
  "yolo_ros/TriggerSelectionRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TriggerSelection-request>)))
  "Returns md5sum for a message object of type '<TriggerSelection-request>"
  "d609001ca88f5f9d939d2ba261852dbd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TriggerSelection-request)))
  "Returns md5sum for a message object of type 'TriggerSelection-request"
  "d609001ca88f5f9d939d2ba261852dbd")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TriggerSelection-request>)))
  "Returns full string definition for message of type '<TriggerSelection-request>"
  (cl:format cl:nil "# Request (empty)~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TriggerSelection-request)))
  "Returns full string definition for message of type 'TriggerSelection-request"
  (cl:format cl:nil "# Request (empty)~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TriggerSelection-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TriggerSelection-request>))
  "Converts a ROS message object to a list"
  (cl:list 'TriggerSelection-request
))
;//! \htmlinclude TriggerSelection-response.msg.html

(cl:defclass <TriggerSelection-response> (roslisp-msg-protocol:ros-message)
  ((object_names
    :reader object_names
    :initarg :object_names
    :type (cl:vector cl:string)
   :initform (cl:make-array 0 :element-type 'cl:string :initial-element ""))
   (object_positions
    :reader object_positions
    :initarg :object_positions
    :type (cl:vector geometry_msgs-msg:Point)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Point :initial-element (cl:make-instance 'geometry_msgs-msg:Point))))
)

(cl:defclass TriggerSelection-response (<TriggerSelection-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TriggerSelection-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TriggerSelection-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name yolo_ros-srv:<TriggerSelection-response> is deprecated: use yolo_ros-srv:TriggerSelection-response instead.")))

(cl:ensure-generic-function 'object_names-val :lambda-list '(m))
(cl:defmethod object_names-val ((m <TriggerSelection-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader yolo_ros-srv:object_names-val is deprecated.  Use yolo_ros-srv:object_names instead.")
  (object_names m))

(cl:ensure-generic-function 'object_positions-val :lambda-list '(m))
(cl:defmethod object_positions-val ((m <TriggerSelection-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader yolo_ros-srv:object_positions-val is deprecated.  Use yolo_ros-srv:object_positions instead.")
  (object_positions m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TriggerSelection-response>) ostream)
  "Serializes a message object of type '<TriggerSelection-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'object_names))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((__ros_str_len (cl:length ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) ele))
   (cl:slot-value msg 'object_names))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'object_positions))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'object_positions))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TriggerSelection-response>) istream)
  "Deserializes a message object of type '<TriggerSelection-response>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'object_names) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'object_names)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:aref vals i) __ros_str_idx) (cl:code-char (cl:read-byte istream))))))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'object_positions) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'object_positions)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Point))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TriggerSelection-response>)))
  "Returns string type for a service object of type '<TriggerSelection-response>"
  "yolo_ros/TriggerSelectionResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TriggerSelection-response)))
  "Returns string type for a service object of type 'TriggerSelection-response"
  "yolo_ros/TriggerSelectionResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TriggerSelection-response>)))
  "Returns md5sum for a message object of type '<TriggerSelection-response>"
  "d609001ca88f5f9d939d2ba261852dbd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TriggerSelection-response)))
  "Returns md5sum for a message object of type 'TriggerSelection-response"
  "d609001ca88f5f9d939d2ba261852dbd")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TriggerSelection-response>)))
  "Returns full string definition for message of type '<TriggerSelection-response>"
  (cl:format cl:nil "# Response~%string[] object_names~%geometry_msgs/Point[] object_positions~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TriggerSelection-response)))
  "Returns full string definition for message of type 'TriggerSelection-response"
  (cl:format cl:nil "# Response~%string[] object_names~%geometry_msgs/Point[] object_positions~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TriggerSelection-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'object_names) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4 (cl:length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'object_positions) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TriggerSelection-response>))
  "Converts a ROS message object to a list"
  (cl:list 'TriggerSelection-response
    (cl:cons ':object_names (object_names msg))
    (cl:cons ':object_positions (object_positions msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'TriggerSelection)))
  'TriggerSelection-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'TriggerSelection)))
  'TriggerSelection-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TriggerSelection)))
  "Returns string type for a service object of type '<TriggerSelection>"
  "yolo_ros/TriggerSelection")