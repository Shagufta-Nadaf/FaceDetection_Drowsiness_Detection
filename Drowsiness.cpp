#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <opencv2/opencv.hpp>

using namespace dlib;
using namespace std;

//---------------------------------------------------------------------------

double euclideanDistance(const point& p1, const point& p2) {
    return sqrt(pow(p2.x() - p1.x(), 2) + pow(p2.y() - p1.y(), 2));
}

// Function to calculate Eye Aspect Ratio (EAR) using Euclidean distance
double calculateEAR(const full_object_detection& shape) {
    double V1 = euclideanDistance(shape.part(38), shape.part(40)); 
    double V2 = euclideanDistance(shape.part(37), shape.part(41)); 
    double H = euclideanDistance(shape.part(39), shape.part(36));
    
    double left = (V1 + V2) / (2.0 * H); // EAR formula
    
    double H1 = euclideanDistance(shape.part(43), shape.part(47)); 
    double H2 = euclideanDistance(shape.part(44), shape.part(46)); 
    double V = euclideanDistance(shape.part(42), shape.part(45));
    
    double right = (H1 + H2) / (2.0 * V);
    
    return (right + left) / 2.0;
}

// Function to calculate Mouth Aspect Ratio (MAR)
double calculateMAR(const full_object_detection& shape) {
    dlib::point p62 = shape.part(62);
    dlib::point p64 = shape.part(64);
    dlib::point p66 = shape.part(66);
    dlib::point p58 = shape.part(58);
    dlib::point p60 = shape.part(60);
    dlib::point p68 = shape.part(68);

    double A = euclideanDistance(p62, p66); // Vertical distance
    double B = euclideanDistance(p64, p60); // Vertical distance
    double C = euclideanDistance(shape.part(61), shape.part(65)); // Horizontal distance

    if (C == 0) {
        cerr << "Error: Horizontal distance C is zero, cannot compute MAR." << endl;
        return -1; // Or some error code
    }

    return (A + B) / (2.0 * C);
}

int main() {
    try {
        // Load face detection and shape prediction models
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize("/home/itstarkenn/opencv_practice/face_dlib/demo/shape_predictor_68_face_landmarks.dat") >> sp;

        // Load input image
        array2d<rgb_pixel> img;

        
        
        
        
        //load_image(img, "/home/itstarkenn/Downloads/download (3).jpeg");
        
 /*       Number of faces detected: 1
Eye Aspect Ratio (EAR): 0.151932
Drowsiness Detected!
Mouth Aspect Ratio (MAR): 1.5

*/
	//load_image(img,"/home/itstarkenn/Downloads/1723403609931 (1).jpg");
	/*Number of faces detected: 1
Eye Aspect Ratio (EAR): 0.333347
Alert!
Mouth Aspect Ratio (MAR): 1.62482
*/
	
	
        // Detect faces in the image
        std::vector<rectangle> dets = detector(img);
        cout << "Number of faces detected: " << dets.size() << endl;

        // Create a window to display the image with landmarks
        image_window win;

        // Draw all bounding boxes first
        win.set_image(img);  
        for (const auto& det : dets) {
            win.add_overlay(det, rgb_pixel(255, 0, 0)); // Red color for bounding box
        }

        // Drowsiness detection threshold
        const double EAR_THRESHOLD = 0.20; // Threshold for detecting closed eyes

        // Iterate through each detected face for landmarks
        for (size_t i = 0; i < dets.size(); ++i) {
            full_object_detection shape = sp(img, dets[i]);

            // Draw landmarks on the image
            for (size_t j = 0; j < shape.num_parts(); ++j) {
                int x = shape.part(j).x();
                int y = shape.part(j).y();
                draw_solid_circle(img, point(x, y), 2, rgb_pixel(0, 255, 0)); // Green color for landmarks
                
                std::string text = std::to_string(j);
                draw_string(img, point(x + 5, y - 5), text, rgb_pixel(255, 255, 255)); // White color for text
            }

            // Calculate EAR and MAR
            double ear = calculateEAR(shape);
            cout << "Eye Aspect Ratio (EAR): " << ear << endl;

            // Check for drowsiness based on EAR
            if (ear < EAR_THRESHOLD) {
                cout << "Drowsiness Detected!" << endl;
            } else {
                cout << "Alert!" << endl;
            }

            double mar = calculateMAR(shape);
            cout << "Mouth Aspect Ratio (MAR): " << mar << endl;

            // Display image with landmarks and bounding boxes
            win.set_image(img);  
            win.add_overlay(render_face_detections(shape));

            cout << "Number of landmarks: " << shape.num_parts() << endl;

            // Print landmark points
            for (size_t j = 0; j < shape.num_parts(); ++j) {
                cout << "Landmark #" << j << ": " << shape.part(j) << endl;
            }
            
        }

        // Save the image with landmarks and bounding boxes
        save_png(img, "output.png");
        cout << "Press Enter to exit..." << endl;
        std::cin.get(); 

    } catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    
    return 0;
}
