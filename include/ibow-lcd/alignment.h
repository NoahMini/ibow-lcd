#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>


namespace ibow_lcd {
    struct AlignmentResult {
        Eigen::Matrix4f transformation;
        int inliers;
        float fitness_score;
    };
    
    AlignmentResult computeCloudTransform(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
        float max_correspondence_distance = 0.05f) 
    {
        AlignmentResult result;
        // Configuración de ICP
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxCorrespondenceDistance(max_correspondence_distance);
        icp.setMaximumIterations(200);
        std::cout << "Fine after setmaxiterations" << std::endl;
        // Ejecutar alineación
        pcl::PointCloud<pcl::PointXYZ> final_cloud;
        icp.align(final_cloud);

        std::cout << "Fine after align" << std::endl;

        if (icp.hasConverged ())
        {
            std::cout << "got into converged" << std::endl;
            // Obtener resultados
            result.transformation = icp.getFinalTransformation();
            result.fitness_score = icp.getFitnessScore();
            // Calcular inliers usando KD-Tree
            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            kdtree.setInputCloud(target);
            int inliers_count = 0;
            for (const auto& point : final_cloud) {
                std::vector<int> indices(1);
                std::vector<float> distances(1);
                if (kdtree.nearestKSearch(point, 1, indices, distances) > 0) {
                    if (distances[0] <= max_correspondence_distance) {
                        inliers_count++;
                    }
                }
            }

            result.inliers = inliers_count;
        }
        else
        {
            std::cout << "didnt converge" << std::endl;
            result.inliers = 0;
            std::cout << "still fine: result inliers equals " << result.inliers << std::endl;
        }
        
        std::cout << "ready to return: inliers = " << result.inliers << std::endl;
        return result;
    }
}