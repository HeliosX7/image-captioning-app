package com.ImageCaptioning;

import java.nio.charset.StandardCharsets;

public class Captioner {
    private static byte[] stringToBytes(String s) {
        return s.getBytes(StandardCharsets.US_ASCII);
    }

    public native void setNumThreads(int numThreads);

    public native int loadModel(String cnn_modelPath, String lstm_modelPath, String weightsPath,String vocublary);

    private native void setMeanWithMeanFile(String meanFile);

    private native void setMeanWithMeanValues(float[] meanValues);

    public native void setScale(float scale);

    public native float[] getConfidenceScore(byte[] data, int width, int height);

    public float[] getConfidenceScore(String imgPath) {
        return getConfidenceScore(stringToBytes(imgPath), 0, 0);
    }

    public native String predictImage(byte[] data, int width, int height, int k);

    public String predictImage(String imgPath, int k) {
        return predictImage(stringToBytes(imgPath), 0, 0, k);
    }

    public String predictImage(String imgPath) {
        return predictImage(imgPath, 1);
    }

    public void setMean(float[] meanValues) {
        setMeanWithMeanValues(meanValues);
    }

    public void setMean(String meanFile) {
        setMeanWithMeanFile(meanFile);
    }
}
