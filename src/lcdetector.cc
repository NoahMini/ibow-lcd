/*
* This file is part of ibow-lcd.
*
* Copyright (C) 2017 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
*
* ibow-lcd is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ibow-lcd is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ibow-lcd. If not, see <http://www.gnu.org/licenses/>.
*/

#include "ibow-lcd/lcdetector.h"
#include "ibow-lcd/alignment.h"

namespace ibow_lcd {

LCDetector::LCDetector(const LCDetectorParams& params) :
      last_lc_island_(-1, 0.0, -1, -1) {
  // Creating the image index
  index_ = std::make_shared<obindex2::ImageIndex>(params.k,
                                                  params.s,
                                                  params.t,
                                                  params.merge_policy,
                                                  params.purge_descriptors,
                                                  params.min_feat_apps);
  // Storing the remaining parameters
  p_ = params.p;
  nndr_ = params.nndr;
  nndr_bf_ = params.nndr_bf;
  ep_dist_ = params.ep_dist;
  conf_prob_ = params.conf_prob;
  min_score_ = params.min_score;
  island_size_ = params.island_size;
  island_offset_ = island_size_ / 2;
  min_inliers_ = params.min_inliers;
  nframes_after_lc_ = params.nframes_after_lc;
  // last_lc_result_.status = LC_NOT_DETECTED;
  min_consecutive_loops_ = params.min_consecutive_loops;
  consecutive_loops_ = 0;
}

LCDetector::~LCDetector() {}

void LCDetector::process(const unsigned image_id,
                         const std::vector<cv::Point3f>& kps,
                         const cv::Mat& descs,
                         std::pair<int, double> &loop_result) {
  
  //FIGURE OUT LOOP_RESULT

  // Storing the keypoints and descriptors
  prev_kps_.push_back(kps);
  prev_descs_.push_back(descs);

  // Adding the current image to the queue to be added in the future
  queue_ids_.push(image_id);
  //std::cout << "Queue ids size: " << queue_ids_.size() << std::endl;

  // Assessing if, at least, p images have arrived
  if (queue_ids_.size() < p_) {
    // result->status = LC_NOT_ENOUGH_IMAGES;
    loop_result.first = -1;
    loop_result.second = 0;
    // last_lc_result_.status = LC_NOT_ENOUGH_IMAGES;
    std::cout << "No loop: Not enough images" << std::endl;
    return;
  }

  // Adding new hypothesis
  unsigned newimg_id = queue_ids_.front();
  queue_ids_.pop();
  auto t_load_start = std::chrono::high_resolution_clock::now();
  
  addImage(newimg_id, prev_kps_[newimg_id], prev_descs_[newimg_id]);

  auto t_load_end = std::chrono::high_resolution_clock::now();
  std::cout << "[Time] AddImage: " << std::chrono::duration_cast<std::chrono::microseconds>(t_load_end - t_load_start).count() << "ms, " << std::endl;

  // Searching similar images in the index
  // Matching the descriptors agains the current visual words
  std::vector<std::vector<cv::DMatch> > matches_feats;

  // Searching the query descriptors against the features
  t_load_start = std::chrono::high_resolution_clock::now();
  
  index_->searchDescriptors(descs, &matches_feats, 2, 64);

  t_load_end = std::chrono::high_resolution_clock::now();
  std::cout << "[Time] SearchDescriptors: " << std::chrono::duration_cast<std::chrono::microseconds>(t_load_end - t_load_start).count() << "ms, " << std::endl;
  // Filtering matches according to the ratio test
  std::vector<cv::DMatch> matches;
  t_load_start = std::chrono::high_resolution_clock::now();
  
  filterMatches(matches_feats, &matches);

  t_load_end = std::chrono::high_resolution_clock::now();
  std::cout << "[Time] FilterMatches: " << std::chrono::duration_cast<std::chrono::microseconds>(t_load_end - t_load_start).count() << "ms, " << std::endl;

  std::vector<obindex2::ImageMatch> image_matches;

  // We look for similar images according to the filtered matches found
  t_load_start = std::chrono::high_resolution_clock::now();
  
  index_->searchImages(descs, matches, &image_matches, true);

  t_load_end = std::chrono::high_resolution_clock::now();
  std::cout << "[Time] SearchImages: " << std::chrono::duration_cast<std::chrono::microseconds>(t_load_end - t_load_start).count() << "ms, " << std::endl;

  // Filtering the resulting image matchings
  std::vector<obindex2::ImageMatch> image_matches_filt;
  t_load_start = std::chrono::high_resolution_clock::now();
  
  filterCandidates(image_matches, &image_matches_filt);

  t_load_end = std::chrono::high_resolution_clock::now();
  std::cout << "[Time] FilterCandidates: " << std::chrono::duration_cast<std::chrono::microseconds>(t_load_end - t_load_start).count() << "ms, " << std::endl;

  std::vector<Island> islands;

  t_load_start = std::chrono::high_resolution_clock::now();
  
  buildIslands(image_matches_filt, &islands);                                       //CHECK MORE THAN JUST THE FIRST ISLAND

  t_load_end = std::chrono::high_resolution_clock::now();
  std::cout << "[Time] BuildIslands: " << std::chrono::duration_cast<std::chrono::microseconds>(t_load_end - t_load_start).count() << "ms, " << std::endl;
  std::cout << "Number of islands: " << islands.size() << std::endl;

  if (!islands.size()) {
    // No resulting islands
    // result->status = LC_NOT_ENOUGH_ISLANDS;
    loop_result.first = -1;
    loop_result.second = 0;
    // last_lc_result_.status = LC_NOT_ENOUGH_ISLANDS;
    std::cout << "No loop: Not enough islands" << std::endl;
    return;
  }

  std::cout << "Resulting Islands:" << std::endl;
  for (unsigned i = 0; i < 5; i++) {
    std::cout << islands[i].toString();
  }

  // Selecting the corresponding island to be processed
  Island island = islands[0];
  std::vector<Island> p_islands;
  t_load_start = std::chrono::high_resolution_clock::now();
  
  getPriorIslands(last_lc_island_, islands, &p_islands);

  t_load_end = std::chrono::high_resolution_clock::now();
  std::cout << "[Time] GetPriorIslands: " << std::chrono::duration_cast<std::chrono::microseconds>(t_load_end - t_load_start).count() << "ms, " << std::endl;
  
  if (p_islands.size()) {
    island = p_islands[0];
  }

  bool overlap = island.overlaps(last_lc_island_);
  last_lc_island_ = island;

  // if () {
  //   consecutive_loops_++;
  // } else {
  //   consecutive_loops_ = 1;
  // }

  unsigned best_img = island.img_id;

  // Assessing the loop
  if (consecutive_loops_ > min_consecutive_loops_ && overlap) {
    // LOOP can be considered as detected
    // result->status = LC_DETECTED;
    loop_result.first = best_img;
    loop_result.second = 0;
    // Store the last result
    // last_lc_result_ = *result;
    std::cout << " Loop detected: Overlap + Enough consecutive loops" << std::endl;
    consecutive_loops_++;
  } else {
    // We obtain the image matchings, since we need them for compute F
    t_load_start = std::chrono::high_resolution_clock::now();
    
    std::vector<cv::DMatch> tmatches;
    std::vector<cv::Point3f> tquery;
    std::vector<cv::Point3f> ttrain;
    ratioMatchingBF(descs, prev_descs_[best_img], &tmatches);
    convertPoints(kps, prev_kps_[best_img], tmatches, &tquery, &ttrain);

    unsigned inliers = 0;
    if (!tquery.empty() && !ttrain.empty()){
      pcl::PointCloud<pcl::PointXYZ> query_cloud;
      query_cloud.points.resize (tquery.size());
      for (size_t i=0; i<tquery.size(); i++) {
            query_cloud.points[i].x = tquery[i].x;
            query_cloud.points[i].y = tquery[i].y;
            query_cloud.points[i].z = tquery[i].z;
      }

      pcl::PointCloud<pcl::PointXYZ> train_cloud;
      train_cloud.points.resize (ttrain.size());
      for (size_t i=0; i<ttrain.size(); i++) {
            train_cloud.points[i].x = ttrain[i].x;
            train_cloud.points[i].y = ttrain[i].y;
            train_cloud.points[i].z = ttrain[i].z;
      }

      ibow_lcd::AlignmentResult result = computeCloudTransform(query_cloud.makeShared(), train_cloud.makeShared());
      std::cout << "got out" << std::endl;
      inliers = result.inliers;
    }
    
    t_load_end = std::chrono::high_resolution_clock::now();
    std::cout << "[Time] CheckForInliers: " << std::chrono::duration_cast<std::chrono::microseconds>(t_load_end - t_load_start).count() << "ms, " << std::endl;
    
    if (inliers > min_inliers_) {
      // LOOP detected
      // result->status = LC_DETECTED;
      loop_result.first = best_img;
      loop_result.second = inliers;
      // Store the last result
      // last_lc_result_ = *result;
      std::cout << " Loop detected: Enough inliers" << std::endl;
      consecutive_loops_++;
    } else {
      // result->status = LC_NOT_ENOUGH_INLIERS;
      loop_result.first = -1;
      loop_result.second = inliers;
      // last_lc_result_.status = LC_NOT_ENOUGH_INLIERS;
      std::cout << " No loop: Not enough inliers" << std::endl;
      consecutive_loops_ = 0;
    }
  }
  // else {
  //   result->status = LC_NOT_DETECTED;
  //   last_lc_result_.status = LC_NOT_DETECTED;
  // }
}

void LCDetector::debug(const unsigned image_id,
             const std::vector<cv::Point3f>& kps,
             const cv::Mat& descs,
             std::ofstream& out_file) {
  auto start = std::chrono::steady_clock::now();
  // Storing the keypoints and descriptors
  prev_kps_.push_back(kps);
  prev_descs_.push_back(descs);

  // Adding the current image to the queue to be added in the future
  queue_ids_.push(image_id);

  // Assessing if, at least, p images have arrived
  if (queue_ids_.size() < p_) {
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    out_file << 0 << "\t";  // min_id
    out_file << 0 << "\t";  // max_id
    out_file << 0 << "\t";  // img_id
    out_file << 0 << "\t";  // overlap
    out_file << 0 << "\t";  // Inliers
    out_file << index_->numDescriptors() << "\t";  // Voc. Size
    out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t";  // Time
    out_file << std::endl;
    return;
  }

  // Adding new hypothesis
  unsigned newimg_id = queue_ids_.front();
  queue_ids_.pop();

  addImage(newimg_id, prev_kps_[newimg_id], prev_descs_[newimg_id]);

  // Searching similar images in the index
  // Matching the descriptors agains the current visual words
  std::vector<std::vector<cv::DMatch> > matches_feats;

  // Searching the query descriptors against the features
  index_->searchDescriptors(descs, &matches_feats, 2, 64);

  // Filtering matches according to the ratio test
  std::vector<cv::DMatch> matches;
  filterMatches(matches_feats, &matches);

  std::vector<obindex2::ImageMatch> image_matches;

  // We look for similar images according to the filtered matches found
  index_->searchImages(descs, matches, &image_matches, true);

  // Filtering the resulting image matchings
  std::vector<obindex2::ImageMatch> image_matches_filt;
  filterCandidates(image_matches, &image_matches_filt);

  std::vector<Island> islands;
  buildIslands(image_matches_filt, &islands);

  if (!islands.size()) {
    // No resulting islands
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    out_file << 0 << "\t";  // min_id
    out_file << 0 << "\t";  // max_id
    out_file << 0 << "\t";  // img_id
    out_file << 0 << "\t";  // overlap
    out_file << 0 << "\t";  // Inliers
    out_file << index_->numDescriptors() << "\t";  // Voc. Size
    out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t";  // Time
    out_file << std::endl;
    return;
  }

  // std::cout << "Resulting Islands:" << std::endl;
  // for (unsigned i = 0; i < islands.size(); i++) {
  //   std::cout << islands[i].toString();
  // }

  // Selecting the corresponding island to be processed
  Island island = islands[0];
  std::vector<Island> p_islands;
  getPriorIslands(last_lc_island_, islands, &p_islands);
  if (p_islands.size()) {
    island = p_islands[0];
  }

  bool overlap = island.overlaps(last_lc_island_);
  last_lc_island_ = island;

  unsigned best_img = island.img_id;

  // We obtain the image matchings, since we need them for compute F
  std::vector<cv::DMatch> tmatches;
  std::vector<cv::Point3f> tquery;
  std::vector<cv::Point3f> ttrain;
  ratioMatchingBF(descs, prev_descs_[best_img], &tmatches);
  convertPoints(kps, prev_kps_[best_img], tmatches, &tquery, &ttrain);
  unsigned inliers = checkEpipolarGeometry(tquery, ttrain);

  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;

  // Writing results
  out_file << island.min_img_id << "\t";          // min_id
  out_file << island.max_img_id << "\t";          // max_id
  out_file << best_img << "\t";                   // img_id
  out_file << overlap << "\t";                    // overlap
  out_file << inliers << "\t";                    // Inliers
  out_file << index_->numDescriptors() << "\t";   // Voc. Size
  out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t";  // Time
  out_file << std::endl;
}

void LCDetector::addImage(const unsigned image_id,
                          const std::vector<cv::Point3f>& kps,
                          const cv::Mat& descs) {
  if (index_->numImages() == 0) {
    // This is the first image that is inserted into the index
    index_->addImage(image_id, kps, descs);
  } else {
    // We have to search the descriptor and filter them before adding descs
    // Matching the descriptors
    std::vector<std::vector<cv::DMatch> > matches_feats;

    // Searching the query descriptors against the features
    index_->searchDescriptors(descs, &matches_feats, 2, 64);

    // Filtering matches according to the ratio test
    std::vector<cv::DMatch> matches;
    filterMatches(matches_feats, &matches);

    // Finally, we add the image taking into account the correct matchings
    index_->addImage(image_id, kps, descs, matches);
  }
}

void LCDetector::filterMatches(
      const std::vector<std::vector<cv::DMatch> >& matches_feats,
      std::vector<cv::DMatch>* matches) {
  // Clearing the current matches vector
  matches->clear();

  // Filtering matches according to the ratio test
  for (unsigned m = 0; m < matches_feats.size(); m++) {
    if (matches_feats[m][0].distance <= matches_feats[m][1].distance * nndr_) {
      matches->push_back(matches_feats[m][0]);
    }
  }
}

void LCDetector::filterCandidates(
      const std::vector<obindex2::ImageMatch>& image_matches,
      std::vector<obindex2::ImageMatch>* image_matches_filt) {
  image_matches_filt->clear();

  double max_score = image_matches[0].score;
  double min_score = image_matches[image_matches.size() - 1].score;

  for (unsigned i = 0; i < image_matches.size(); i++) {
    // Computing the new score
    double new_score = (image_matches[i].score - min_score) /
                       (max_score - min_score);
    // Assessing if this image match is higher than a threshold
    if (new_score > min_score_) {
      obindex2::ImageMatch match = image_matches[i];
      match.score = new_score;
      image_matches_filt->push_back(match);
    } else {
      break;
    }
  }
}

void LCDetector::buildIslands(
      const std::vector<obindex2::ImageMatch>& image_matches,
      std::vector<Island>* islands) {
  islands->clear();

  // We process each of the resulting image matchings
  for (unsigned i = 0; i < image_matches.size(); i++) {
    // Getting information about this match
    unsigned curr_img_id = static_cast<unsigned>(image_matches[i].image_id);
    double curr_score = image_matches[i].score;

    // Theoretical island limits
    unsigned min_id = static_cast<unsigned>
                              (std::max((int)curr_img_id - (int)island_offset_,
                               0));
    unsigned max_id = curr_img_id + island_offset_;

    // We search for the closest island
    bool found = false;
    for (unsigned j = 0; j < islands->size(); j++) {
      if (islands->at(j).fits(curr_img_id)) {
        islands->at(j).incrementScore(curr_score);
        found = true;
        break;
      } else {
        // We adjust the limits of a future island
        islands->at(j).adjustLimits(curr_img_id, &min_id, &max_id);
      }
    }

    // Creating a new island if required
    if (!found) {
      Island new_island(curr_img_id,
                        curr_score,
                        min_id,
                        max_id);
      islands->push_back(new_island);
    }
  }

  // Normalizing the final scores according to the number of images
  for (unsigned j = 0; j < islands->size(); j++) {
    islands->at(j).normalizeScore();
  }

  std::sort(islands->begin(), islands->end());
}

void LCDetector::getPriorIslands(
      const Island& island,
      const std::vector<Island>& islands,
      std::vector<Island>* p_islands) {
  p_islands->clear();

  // We search for overlapping islands
  for (unsigned i = 0; i < islands.size(); i++) {
    Island tisl = islands[i];
    if (island.overlaps(tisl)) {
      p_islands->push_back(tisl);
    }
  }
}

unsigned LCDetector::checkEpipolarGeometry(                                               //Apply icp with pointclouds
                                      const std::vector<cv::Point3f>& query,
                                      const std::vector<cv::Point3f>& train) {
  std::vector<uchar> inliers(query.size(), 0);
  cv::Mat aff(3,4,CV_64F);
  if (query.size() > 7) {
    int ret =
      cv::estimateAffine3D(
        query, train,      // Matching points
        aff,                                  // output transformation matrix between the 3d point sets
        inliers,                            // Output vector indicating which points are inliers (1-inlier, 0-outlier). 
        ep_dist_,                                 // Distance to epipolar line
        conf_prob_);                              // Confidence probability
  }

  // Extract the surviving (inliers) matches
  auto it = inliers.begin();
  unsigned total_inliers = 0;
  for (; it != inliers.end(); it++) {
    if (*it)
      total_inliers++;
  }

  return total_inliers;
}

void LCDetector::ratioMatchingBF(const cv::Mat& query,
                                 const cv::Mat& train,
                                 std::vector<cv::DMatch>* matches) {
  matches->clear();
  cv::BFMatcher matcher(cv::NORM_HAMMING);

  // Matching descriptors
  std::vector<std::vector<cv::DMatch> > matches12;
  matcher.knnMatch(query, train, matches12, 2);

  // Filtering the resulting matchings according to the given ratio
  for (unsigned m = 0; m < matches12.size(); m++) {
    if (matches12[m][0].distance <= matches12[m][1].distance * nndr_bf_) {
      matches->push_back(matches12[m][0]);
    }
  }
}

void LCDetector::convertPoints(const std::vector<cv::Point3f>& query_kps, // kps
                               const std::vector<cv::Point3f>& train_kps, // prev_kps[best_img]
                               const std::vector<cv::DMatch>& matches,
                               std::vector<cv::Point3f>* query,
                               std::vector<cv::Point3f>* train) {
  query->clear();
  train->clear();
  for (auto it = matches.begin(); it != matches.end(); it++) {
    // Get the position of query keypoints
    float x = query_kps[it->queryIdx].x;
    float y = query_kps[it->queryIdx].y;
    float z = query_kps[it->queryIdx].z;
    query->push_back(cv::Point3f(x, y, z));

    // Get the position of train keypoints
    x = train_kps[it->trainIdx].x;
    y = train_kps[it->trainIdx].y;
    z = train_kps[it->trainIdx].z;
    train->push_back(cv::Point3f(x, y, z));
  }
}

}  // namespace ibow_lcd
