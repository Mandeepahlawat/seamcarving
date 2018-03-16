
#include "sc.h"

using namespace cv;
using namespace std;

Mat image_gradient(Mat& image){
  Mat image_blur;
  Mat image_gray;
  Mat grad, energy_image;
  int scale = 1;
  int delta = 0;
  int depth = CV_16S;

  GaussianBlur(image, image_blur, Size(3,3), 0, 0, BORDER_DEFAULT);

  cvtColor(image_blur, image_gray, CV_BGR2GRAY);

  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  Sobel(image_gray, grad_x, depth, 1, 0, 1, scale, delta, BORDER_DEFAULT);
  // Scharr(image_gray, grad_x, depth, 1, 0, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);

  Sobel(image_gray, grad_y, depth, 0, 1, 1, scale, delta, BORDER_DEFAULT);
  // Scharr(image_gray, grad_y, depth, 0, 1, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(grad_y, abs_grad_y);

  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

  // convert the default values to double precision
  grad.convertTo(energy_image, CV_64F, 1.0/255.0);

  // namedWindow("Energy Image", CV_WINDOW_AUTOSIZE); imshow("Energy Image", energy_image);

  return energy_image;
}

Mat vertical_energy_matrix(Mat& image){
  int rowsize = image.rows;
  int colsize = image.cols;
  double p1, p2, p3;

  Mat energy_matrix = Mat(rowsize, colsize, CV_64F, double(0));
  image.row(0).copyTo(energy_matrix.row(0));

  for(int i = 1; i < rowsize; i++){
    for(int j = 0; j < colsize; j++){
      // cout << "vertical matrix"<<endl;
      p1 = energy_matrix.at<double>(i-1, max(j-1, 0));  // upper left pixel, max to handle the edge case
      p2 = energy_matrix.at<double>(i-1, j);            // just above pixel
      p3 = energy_matrix.at<double>(i-1, min(j+1,colsize-1)); // upper right pixel, min to handle the edge case
      energy_matrix.at<double>(i,j) = image.at<double>(i,j) + std::min(p1,min(p2,p3));  // add current pixel and min of above pixels 
    }
  }
  return energy_matrix;
}

Mat horizontal_energy_matrix(Mat& image){
  int rowsize = image.rows;
  int colsize = image.cols;
  double p1, p2, p3;

  Mat energy_matrix = Mat(rowsize, colsize, CV_64F, double(0));
  image.col(0).copyTo(energy_matrix.col(0));
  // cout << "size of energy_matrix: "<< energy_matrix.size() << endl;

  // cout << "rowsize: " << rowsize << endl;
  // cout << "colsize: " << colsize << endl;

  // double im1 = energy_matrix.at<double>(968,96);
  // cout << "oh fuck even this is working: " << im1 <<endl;

  for(int i = 1; i < colsize; i++){
    for(int j = 0; j < rowsize; j++){
      // cout << "i: " << i << "j: "<< j <<endl;
      // cout << "energy_matrix: " << energy_matrix.at<double>(i,j) <<endl;
      // cout << "energy_matrix: " << energy_matrix.at<float>(i,j) <<endl;

      p1 = energy_matrix.at<double>(max(j-1, 0), i - 1);  // right upper pixel, max to handle the edge case
      // cout << "p1: " << p1 << endl;
      p2 = energy_matrix.at<double>(j, i - 1);            // just right pixel
      // cout << "p2: " << p2 << endl;
      p3 = energy_matrix.at<double>(min(j+1, rowsize-1), i - 1); // right down pixel, min to handle the edge case
      // cout << "p3: " << p3 << endl;

      // double im1 = energy_matrix.at<double>(i,j);
      // cout << "oh fuck even this is working: " << im1 <<endl;

      energy_matrix.at<double>(j,i) = image.at<double>(j,i) + std::min(p1,min(p2,p3));  // add current pixel and min of above pixels 
      // cout << "energy_matrix: " << energy_matrix.at<double>(i,j) <<endl;
    }
  }

  // cout << "============================= energy_matrix ==========================="<<endl;
  // cout << energy_matrix << endl;


  // cout << "horizontal matrix done"<<endl;
  
  // namedWindow("Cumulative Energy Map", CV_WINDOW_AUTOSIZE); imshow("Cumulative Energy Map", color_cumulative_energy_map);

  return energy_matrix;
}

bool seam_carving(Mat& in_image, int new_width, int new_height, Mat& out_image){

    // some sanity checks
    // Check 1 -> new_width <= in_image.cols
    if(new_width>in_image.cols){
        cout<<"Invalid request!!! new_width has to be smaller than the current size!"<<endl;
        return false;
    }
    if(new_height>in_image.rows){
        cout<<"Invalid request!!! ne_height has to be smaller than the current size!"<<endl;
        return false;
    }
    
    if(new_width<=0){
        cout<<"Invalid request!!! new_width has to be positive!"<<endl;
        return false;

    }
    
    if(new_height<=0){
        cout<<"Invalid request!!! new_height has to be positive!"<<endl;
        return false;
        
    }

    
    return seam_carving_trivial(in_image, new_width, new_height, out_image);
}


// seam carves by removing trivial seams
bool seam_carving_trivial(Mat& in_image, int new_width, int new_height, Mat& out_image){

    Mat iimage = in_image.clone();
    Mat oimage = in_image.clone();

    // Mat imagegradient = image_gradient(in_image);

    // Mat v_eg_matrix = vertical_energy_matrix(imagegradient);
    // cout<< "======================================================================="<<endl;
    // Mat hz_eg_matrix = horizontal_energy_matrix(imagegradient);
    // cout<< "======================================================================="<<endl;

    // int count = 0;

    // cout << "=================== new height: "<< new_height<<" ============================="<<endl;

    // cout << "=================== new width: "<< new_width<<" ============================="<<endl;

    // while(iimage.rows!=new_height || iimage.cols!=new_width){
      // cout << "=================== transfored height: "<< iimage.rows<<" ============================="<<endl;

      // cout << "=================== transfored width: "<< iimage.cols<<" ============================="<<endl;

      // count = count+1;
      // horizontal seam if needed
      // if(iimage.rows>new_height){
          // cout << "=================== horizontal SEAM:" <<"============================="<<endl;
          // reduce_vertical_seam_trivial(iimage, oimage, v_eg_matrix);
          // Mat imagegradient = image_gradient(iimage);
          // Mat hz_eg_matrix = horizontal_energy_matrix(imagegradient);

          // reduce_horizontal_seam_trivial(iimage, oimage, hz_eg_matrix);
          // cout << "=================== height count finished============================="<<endl;

          // iimage = oimage.clone();
      // }
      
      // if(iimage.cols>new_width){
          // cout << "=================== vertical SEAM:" <<"============================="<<endl;
          // Mat imagegradient = image_gradient(iimage);
          // Mat v_eg_matrix = vertical_energy_matrix(imagegradient);

          // reduce_vertical_seam_trivial(iimage, oimage, v_eg_matrix);
          // reduce_horizontal_seam_trivial(iimage, oimage, hz_eg_matrix);
          // cout << "=================== width count finished============================="<<endl;
          // iimage = oimage.clone();
      // }
      // cout << "=================== transfored row: "<< iimage.rows<<" ============================="<<endl;
      // cout << "=================== transfored cols: "<< iimage.cols<<" ============================="<<endl;
    // }

    // cout << "=================== height and width equal============================="<<endl;
    
    // out_image = oimage.clone();
    // int iterations = max((iimage.rows - new_height), iimage.cols - new_width);
    int height_iterations = iimage.rows - new_height;
    int width_iterations = iimage.cols - new_width;

    for(int i = 0; i < width_iterations; ++i){
      Mat imagegradient = image_gradient(iimage);
      Mat v_eg_matrix = vertical_energy_matrix(imagegradient);
      
      reduce_vertical_seam_trivial(iimage, oimage, v_eg_matrix);
      cout << "Reducing width: " << i << "/" << width_iterations << endl;
      iimage = oimage.clone();
    }

    for(int i = 0; i < height_iterations; ++i){
      Mat imagegradient = image_gradient(iimage);
      Mat hz_eg_matrix = horizontal_energy_matrix(imagegradient);

      reduce_horizontal_seam_trivial(iimage, oimage, hz_eg_matrix);
      iimage = oimage.clone();
      cout << "Reducing height: " << i << "/" << height_iterations << endl;
    }
    out_image = oimage.clone();

    return true;
}

// // horizontl trivial seam is a seam through the center of the image
bool reduce_horizontal_seam_trivial(Mat& in_image, Mat& out_image, Mat& energy_matrix){
    int egm_row = energy_matrix.rows;
    int egm_col = energy_matrix.cols;

    double p1,p2,p3;
    vector<int> seam_path;
    double min_val, max_val;
    Point min_point,max_point;
    int min_energy_index = 0;
    int offset = 0;

    seam_path.resize(egm_col);

    minMaxLoc(energy_matrix.col(egm_col - 1), &min_val, &max_val, &min_point, &max_point);
    
    min_energy_index = min_point.y;
    seam_path[egm_col - 1] = min_energy_index;

    for(int i = egm_col - 2; i >= 0; i--){
      // cout << "horizontal seam trivial inside the looppppppppppppppppppp" << endl;
      p1 = energy_matrix.at<double>(max(min_energy_index - 1,0),i); // edge case, right top most row
      p2 = energy_matrix.at<double>(min_energy_index,i);              // pixel immediate right
      p3 = energy_matrix.at<double>(min(min_energy_index + 1, egm_row - 1),i); // edge case, right downmost row
      
      if(min(p1,p2) > p3){
        offset = 1;
      }
      else if(min(p2,p3) > p1){
        offset = -1;
      }
      else if(min(p1,p3) > p2){
        offset = 0;
      }
      // cout << "offset: " << offset << endl;

      min_energy_index = min_energy_index + offset;
      min_energy_index = min(max(min_energy_index,0),egm_row - 1); // edge case
      seam_path[i] = min_energy_index;

    }

    int rows = in_image.rows;
    int cols = in_image.cols;

    Mat dummy_column(1, 1, CV_8UC3, Vec3b(0, 0, 0));

    for (int i = 0; i < cols; i++) {
      Mat new_column;
      Mat upper_matrix = in_image.colRange(i, i+1).rowRange(seam_path[i] + 1, rows);
      Mat lower_matrix = in_image.colRange(i, i+1).rowRange(0, seam_path[i]);
  

      if(!upper_matrix.empty() && !lower_matrix.empty()){
        vconcat(lower_matrix, upper_matrix, new_column);
        vconcat(new_column, dummy_column, new_column);
      }
      else{
        if(upper_matrix.empty()){
          vconcat(lower_matrix, dummy_column, new_column);
        }
        else if(lower_matrix.empty()){
          vconcat(upper_matrix, dummy_column, new_column);
        }
      }
      new_column.copyTo(out_image.col(i));
    }

    // cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<endl;

    out_image = out_image.rowRange(0, rows - 1);
    // out_image = in_image.clone();

    return true;
}

// vertical trivial seam is a seam through the center of the image
bool reduce_vertical_seam_trivial(Mat& in_image, Mat& out_image, Mat& energy_matrix){
    int egm_row = energy_matrix.rows;
    int egm_col = energy_matrix.cols;

    double p1,p2,p3;
    vector<int> seam_path;
    double min_val, max_val;
    Point min_point,max_point;
    // int min_energy_index = 0;
    int offset = 0;


    seam_path.resize(egm_row);
    
    // cout << "vertical seam trivial" << endl;
    minMaxLoc(energy_matrix.row(egm_row - 1), &min_val, &max_val, &min_point, &max_point);
    int min_energy_index = min_point.x;
    seam_path[egm_row - 1] = min_energy_index;

    for(int i = egm_row - 2; i >= 0; i--){
      // cout << "vertical seam trivial inside the looppppppppppppppppppp" << endl;
      p1 = energy_matrix.at<double>(i,max(min_energy_index - 1, 0)); // edge case, left most column
      p2 = energy_matrix.at<double>(i,min_energy_index);              // pixel above
      p3 = energy_matrix.at<double>(i,min(min_energy_index + 1, egm_col - 1)); // edge case, right most column
      
      if(min(p1,p2) > p3){
        offset = 1;
      }
      else if(min(p2,p3) > p1){
        offset = -1;
      }
      else if(min(p1,p3) > p2){
        offset = 0;
      }
      min_energy_index = min_energy_index + offset;
      min_energy_index = min(max(min_energy_index,0),egm_col - 1); // edge case
      seam_path[i] = min_energy_index;
    }

    // dimensions of the new image
    int rows = in_image.rows;
    int cols = in_image.cols;

    Mat dummy_row(1, 1, CV_8UC3, Vec3b(0, 0, 0));

    for (int i = 0; i < rows; i++) {

      Mat new_row;
      Mat left_matrix = in_image.rowRange(i, i + 1).colRange(0, seam_path[i]);
      Mat right_matrix = in_image.rowRange(i, i + 1).colRange(seam_path[i] + 1, cols);


      if(!left_matrix.empty() && !right_matrix.empty()){
        hconcat(left_matrix, right_matrix, new_row);
        hconcat(new_row, dummy_row, new_row);
      }
      else{
        if(left_matrix.empty()){
          hconcat(right_matrix, dummy_row, new_row);
        }
        else if(right_matrix.empty()){
          hconcat(left_matrix, dummy_row, new_row);
        }
      }
      
      new_row.copyTo(out_image.row(i));
    }

    out_image = out_image.colRange(0, cols - 1);

    
    return true;
}
