// Auto-generated. Do not edit!

// (in-package yolo_ros.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class TriggerSelectionRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TriggerSelectionRequest
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TriggerSelectionRequest
    let len;
    let data = new TriggerSelectionRequest(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'yolo_ros/TriggerSelectionRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd41d8cd98f00b204e9800998ecf8427e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Request (empty)
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new TriggerSelectionRequest(null);
    return resolved;
    }
};

class TriggerSelectionResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.object_names = null;
      this.object_positions = null;
    }
    else {
      if (initObj.hasOwnProperty('object_names')) {
        this.object_names = initObj.object_names
      }
      else {
        this.object_names = [];
      }
      if (initObj.hasOwnProperty('object_positions')) {
        this.object_positions = initObj.object_positions
      }
      else {
        this.object_positions = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TriggerSelectionResponse
    // Serialize message field [object_names]
    bufferOffset = _arraySerializer.string(obj.object_names, buffer, bufferOffset, null);
    // Serialize message field [object_positions]
    // Serialize the length for message field [object_positions]
    bufferOffset = _serializer.uint32(obj.object_positions.length, buffer, bufferOffset);
    obj.object_positions.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Point.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TriggerSelectionResponse
    let len;
    let data = new TriggerSelectionResponse(null);
    // Deserialize message field [object_names]
    data.object_names = _arrayDeserializer.string(buffer, bufferOffset, null)
    // Deserialize message field [object_positions]
    // Deserialize array length for message field [object_positions]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.object_positions = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.object_positions[i] = geometry_msgs.msg.Point.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.object_names.forEach((val) => {
      length += 4 + _getByteLength(val);
    });
    length += 24 * object.object_positions.length;
    return length + 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'yolo_ros/TriggerSelectionResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd609001ca88f5f9d939d2ba261852dbd';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Response
    string[] object_names
    geometry_msgs/Point[] object_positions
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new TriggerSelectionResponse(null);
    if (msg.object_names !== undefined) {
      resolved.object_names = msg.object_names;
    }
    else {
      resolved.object_names = []
    }

    if (msg.object_positions !== undefined) {
      resolved.object_positions = new Array(msg.object_positions.length);
      for (let i = 0; i < resolved.object_positions.length; ++i) {
        resolved.object_positions[i] = geometry_msgs.msg.Point.Resolve(msg.object_positions[i]);
      }
    }
    else {
      resolved.object_positions = []
    }

    return resolved;
    }
};

module.exports = {
  Request: TriggerSelectionRequest,
  Response: TriggerSelectionResponse,
  md5sum() { return 'd609001ca88f5f9d939d2ba261852dbd'; },
  datatype() { return 'yolo_ros/TriggerSelection'; }
};
