import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

    public static void main(String[] args) throws IOException {
        String globPattern = "glob:**/*.xml";
        String folderPATH = "C:\\Users\\ChooWilson\\Desktop\\invoice\\train";
        getXmlFiles(globPattern, folderPATH);
        System.out.println("Total number of xml files: " + xmlFiles.size());
        all_boxes = xml2boxes();
        System.out.println("NDArray shape: " + Arrays.toString(all_boxes.shape()));
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
