# Copyright (C) 2024 Intel Corporation , Extended by Olive Robotics GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from geti_sdk.deployment import Deployment
from geti_sdk.utils import show_image_with_annotation_scene

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/olive/camera/id01/image/compressed',
            self.image_callback,
            rclpy.qos.qos_profile_sensor_data)
        
        self.publisher_ = self.create_publisher(Float32, '/olive/servo/id23/goal/velocity', 10)
        self.image_publisher_ = self.create_publisher(CompressedImage, 'olive/camera/id01/geti/image/compressed', 10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.deployment = Deployment.from_folder("../deployment")
        self.deployment.load_inference_models(device="CPU")
        self.one_time = 0
        self.label_index = 0

    def image_callback(self, msg):
            self.one_time = self.one_time + 1
            if self.one_time == 5:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                #cv2.imshow("Live Camera Feed", cv_image)
                #key = cv2.waitKey(1)
                image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                #print('new image converted')
                prediction = self.deployment.infer(image_rgb)
                #print(f"Prediction result: {prediction}")
                # Draw annotations on the image
                for annotation in prediction.annotations:
                    for label in annotation.labels:
                        x = annotation.shape.x
                        y = annotation.shape.y
                        width = annotation.shape.width
                        height = annotation.shape.height
                        label_name = label.name
                        # Determine label index
                        if label_name == 'Wall':
                            label_index = 0
                        elif label_name == 'Beer':
                            label_index = 1
                        elif label_name == 'Duck':
                            label_index = -1

                        color = tuple(int(label.color[i:i+2], 16) for i in (1, 3, 5))  # Convert hex color to BGR
                        print("draw" + label_name)
                        cv2.rectangle(cv_image, (0, 0), (100, 100), color, 4)
                        # Put label text
                        cv2.putText(cv_image, label_name, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                #show_image_with_annotation_scene(image_rgb, prediction)
                #cv2.imshow("Live Camera Feed", cv_image)
                #key = cv2.waitKey(1)

                # Publish label index
                label_index_msg = Float32()
                label_index_msg.data = float(label_index)
                self.publisher_.publish(label_index_msg)

                 # Convert annotated image to compressed image message
                _, buffer = cv2.imencode('.jpg', cv_image)
                compressed_image_msg = CompressedImage()
                compressed_image_msg.header.stamp = self.get_clock().now().to_msg()
                compressed_image_msg.format = "jpeg"
                compressed_image_msg.data = buffer.tobytes()
                self.image_publisher_.publish(compressed_image_msg)

                # Draw annotations on the image
                print('finish')
                self.one_time = 0


def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



