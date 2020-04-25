import numpy as np

class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        # n = number of boxes/examples
        # k = no. of clusters
        n = boxes.shape[0]
        k = self.cluster_number

        # box width * box height
        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    # only used to calculate the accuracy by iou
    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    # kmeans function
    def kmeans(self, boxes, k, dist=np.median):
        # boxes = numpy array of all boxes
        #  k = number of clusters.
        # boxes.shape[0] = 5028 number of rows/example
        # boxes.shape[1] = 2 number of column
        box_number = boxes.shape[0]
        # distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        # if you want to seed
        np.random.seed(000)
        #  box_num = 5082, k = 9
        # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html

        # randomly init 9 centroids
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:
            # calculate distance by using 1 - box IOU
            # usually Kmeans use euclidian distance
            distances = 1 - self.iou(boxes, clusters)

            # current_nearest contains the best score of which cluster each w,h belongs to
            current_nearest = np.argmin(distances, axis=1)
            # numpy . all ?
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            # do n times for number of cluster e.g. 9 clusters
            for cluster in range(k):
                # loop through all cluster
                # use the distance formula, author use numpy.median
                #  numpy median, sort and get the term in the middle, if num elements is even, get mean
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            # last_nearest is not the same as current_nearest, do update
            last_nearest = current_nearest

        return clusters

    # Write results to a text file containing the anchors
    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            # collect all the (Width,height) for all objects
            # append array of [width,height]
            #width = xmax - xmin
            #height = ymax - ymin
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        #convert list in a list into a numpy array
        result = np.array(dataSet)
        f.close()
        return result

    # last function
    def txt2clusters(self):
        # main thing to do
        # all_boxes is a numpy array that stores all the w,h information
        all_boxes = self.txt2boxes()
        # perform kmeans by using the w,h info of all boxes
        result = self.kmeans(all_boxes, k=self.cluster_number)
        # what is lexsort??
        result = result[np.lexsort(result.T[0, None])]
        # write to text file
        self.result2txt(result)
        # print results
        print("K anchors:\n {}".format(result))
        # print the accuracy by avg_iou
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))

if __name__ == "__main__":
    cluster_number = 9
    filename = "invoice.txt"
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()
