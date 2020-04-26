import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.nd4j.linalg.ops.transforms.Transforms.min;


// Reference to https://github.com/eclipse/deeplearning4j/blob/master/datavec/datavec-data/datavec-data-image/src/main/java/org/datavec/image/recordreader/objdetect/impl/VocLabelProvider.java
// Reference https://javapapers.com/java/glob-with-java-nio/

public class GenerateAnchorBox {

    private static final String OBJECT_START_TAG = "<object>";
    private static final String OBJECT_END_TAG = "</object>";
    private static final String XMIN_TAG = "<xmin>";
    private static final String YMIN_TAG = "<ymin>";
    private static final String XMAX_TAG = "<xmax>";
    private static final String YMAX_TAG = "<ymax>";
    private static List<String> xmlFiles = new ArrayList<>();
    private static INDArray all_boxes = Nd4j.empty();
    private static int numClusters = 9;

    public static void main(String[] args) throws IOException {
        String globPattern = "glob:**/*.xml";
        String folderPATH = "C:\\Users\\ChooWilson\\Desktop\\invoice\\train";
        getXmlFiles(globPattern, folderPATH);
        System.out.println("Total number of xml files: " + xmlFiles.size());
        all_boxes = xml2boxes();

        INDArray results = KMeans(all_boxes, numClusters);
    }

    private static INDArray KMeans(INDArray boxes, int k) {
        INDArray allboxes = boxes;
        int box_number = (int) boxes.shape()[0];
//        System.out.println(box_number);
        INDArray last_nearest = Nd4j.zeros(new int[]{box_number,});
//        System.out.println(last_nearest);

//        Generates a random sample from all of the boxes
        Random rng = new Random();
        INDArray clusters = Nd4j.empty();
        for (int i = 0; i < k; i++) {
            int num = rng.nextInt(box_number);
            INDArray sample = allboxes.getRow(num);
            sample = sample.reshape(new int[]{1, 2});
            if (clusters == Nd4j.empty()) {
                clusters = sample;
            } else {
                clusters = Nd4j.concat(0, clusters, sample);
            }
        }
//        System.out.println(clusters);
//        System.out.println(Arrays.toString(clusters.shape()));

//        INDArray results = iou(allboxes, clusters);
//        System.out.println(results);
//        System.out.println(Arrays.toString(results.shape()));

        while (true) {
            INDArray ones = Nd4j.ones(new int[]{box_number, k});
            INDArray distances = ones.sub(iou(allboxes, clusters));
//            System.out.println(distances);
            INDArray current_nearest = Nd4j.argMin(distances, 1).castTo(DataType.FLOAT);

            System.out.println(last_nearest.eq(current_nearest));
//                .all() is to account for all booleans
            System.out.println(last_nearest.eq(current_nearest).all());
            if (last_nearest.eq(current_nearest).all()) {
                System.out.println("exit");
                break;
            }

//             it never convergeces..
            for (int cluster = 0; cluster < k; cluster++) {
                INDArray med = boxes.get(current_nearest.eq(cluster)).median(0);
                System.out.println(med);
                clusters.putRow(cluster, med);
                System.out.println(clusters);
            }
            last_nearest = current_nearest;
        }
        return clusters;
    }

    private static INDArray iou(INDArray boxes, INDArray clusters) {

        int n = (int) boxes.shape()[0];
        int k = numClusters;

//        Get the areas for all annotation boxes first
        INDArray allBoxWidths = all_boxes.slice(0, 1);
        INDArray allBoxHeights = all_boxes.slice(1, 1);
        INDArray allBoxArea = allBoxWidths.mul(allBoxHeights);
        allBoxArea = allBoxArea.repeat(0, k);
        allBoxArea = allBoxArea.reshape(new int[]{n, k});

//        Get the areas for all cluster boxes first
        INDArray allClusterWidths = clusters.slice(0, 1);
        INDArray allClusterHeights = clusters.slice(1, 1);
        INDArray allClusterArea = allClusterWidths.mul(allClusterHeights);
        allClusterArea = Nd4j.tile(allClusterArea, n);
        ////        this line might be wrong
        allClusterArea = allClusterArea.reshape(new int[]{n, k});

        INDArray box_w_matrix = allBoxWidths.repeat(0, k).reshape(new int[]{n, k});
        INDArray cluster_w_matrix = Nd4j.tile(allClusterWidths, n).reshape(new int[]{n, k});
        INDArray min_w_matrix = min(cluster_w_matrix, box_w_matrix);

        INDArray box_h_matrix = allBoxHeights.repeat(0, k).reshape(new int[]{n, k});
        INDArray cluster_h_matrix = Nd4j.tile(allClusterHeights, n).reshape(new int[]{n, k});
        INDArray min_h_matrix = min(cluster_h_matrix, box_h_matrix);

        INDArray intersect = min_w_matrix.mul(min_h_matrix);
        INDArray union = allBoxArea.add(allClusterArea).sub(intersect);
        INDArray i_o_u = intersect.div(union);
        return i_o_u;
    }

    private static INDArray xml2boxes() {
        for (String xmlPATH : xmlFiles) {
            File xmlFile = new File(xmlPATH);
            if (!xmlFile.exists()) {
                throw new IllegalStateException("Could not find XML file.");
            }

            String xmlContent;
            try {
                xmlContent = FileUtils.readFileToString(xmlFile, "UTF-8");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            String[] lines = xmlContent.split("\n");
            for (int i = 0; i < lines.length; i++) {
                if (!lines[i].contains(OBJECT_START_TAG)) {
                    continue;
                }
                //        Initialize to min value instead of zero
                int xmin = Integer.MIN_VALUE;
                int ymin = Integer.MIN_VALUE;
                int xmax = Integer.MIN_VALUE;
                int ymax = Integer.MIN_VALUE;
                while (!lines[i].contains(OBJECT_END_TAG)) {
                    if (xmin == Integer.MIN_VALUE && lines[i].contains(XMIN_TAG)) {
                        xmin = extractAndParse(lines[i]);
                        i++;
                        continue;
                    }
                    if (ymin == Integer.MIN_VALUE && lines[i].contains(YMIN_TAG)) {
                        ymin = extractAndParse(lines[i]);
                        i++;
                        continue;
                    }
                    if (xmax == Integer.MIN_VALUE && lines[i].contains(XMAX_TAG)) {
                        xmax = extractAndParse(lines[i]);
                        i++;
                        continue;
                    }
                    if (ymax == Integer.MIN_VALUE && lines[i].contains(YMAX_TAG)) {
                        ymax = extractAndParse(lines[i]);
                        i++;
                        continue;
                    }
                    i++;
                }
                int width = xmax - xmin;
                int height = ymax - ymin;
                INDArray width_height = Nd4j.create(new float[]{width, height}, new int[]{1, 2});
                if (all_boxes == Nd4j.empty()) {
                    all_boxes = width_height;
                } else {
                    all_boxes = Nd4j.concat(0, all_boxes, width_height);
                }
            }
        }
        return all_boxes;
    }

    private static int extractAndParse(String line) {
        int idxStartName = line.indexOf('>') + 1;
        int idxEndName = line.lastIndexOf('<');
        String substring = line.substring(idxStartName, idxEndName);
        return Integer.parseInt(substring);
    }

    // Get a all the xml files in the specified folder and store them to a list of strings
    private static void getXmlFiles(String glob, String folder) throws IOException {

        final PathMatcher pathMatcher = FileSystems.getDefault().getPathMatcher(glob);
        Files.walkFileTree(Paths.get(folder), new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path path, BasicFileAttributes attrs) {
                if (pathMatcher.matches(path)) {
//                    System.out.println(path);
                    xmlFiles.add(path.toString());
                }
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFileFailed(Path file, IOException exc) {
                return FileVisitResult.CONTINUE;
            }
        });
    }
}
