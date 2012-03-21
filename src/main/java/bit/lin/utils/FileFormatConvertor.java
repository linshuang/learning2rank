package bit.lin.utils;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * this class is used to pre-process files from microsoft learning to rank data
 * set
 */
public class FileFormatConvertor {

	public void m2j() throws IOException {
		String[] fname = { "/test.txt", "/train.txt", "/vali.txt" };
		for (int i = 1; i <= 5; i++)
			for (String name : fname) {
				// random access file
				RandomAccessFile raf4r = new RandomAccessFile(
						"/home/lins/data/Learning to rank/10k/Fold" + i + name,
						"r");
				RandomAccessFile raf4w = new RandomAccessFile(
						"/home/lins/data/Learning to rank/10k4j/Fold" + i
								+ name, "rw");
				// from reader to writer
				while (true) {
					String tmp = raf4r.readLine();
					if (tmp == null)
						break;
					tmp = tmp.replaceAll("\\s[0-9|qid]*:", ";");
					System.out.println(tmp);
					raf4w.writeBytes(tmp + "\n");
				}

				// close the accessor
				raf4r.close();
				raf4w.close();
			}
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		FileFormatConvertor convertor = new FileFormatConvertor();
		convertor.m2j();
//		String tmp = "2 qid:13 1:2 2:0 3:2 4:1 5:2 6:1 7:0 8:1 9:0.50000 10:1 11:31 12:0 13:11 14:7 15:49 16:6.553125 17:15.011174 18:12.950828 19:14.369216 20:6.550869 21:4 22:0 23:2 24:1 25:7 26:2 27:0 28:1 29:0 30:3 31:2 32:0 33:1 34:1 35:4 36:2 37:0 38:1 39:0.50000 40:3.50000 41:0 42:0 43:0 44:0.25000 45:0.25000 46:0.129032 47:0 48:0.181818 49:0.142857 50:0.142857 51:0.064516 52:0 53:0.090909 54:0 55:0.061224 56:0.064516 57:0 58:0.090909 59:0.142857 60:0.081633 61:0.064516 62:0 63:0.090909 64:0.071429 65:0.071429 66:0 67:0 68:0 69:0.005102 70:0.000104 71:13.106251 72:0 73:12.950828 74:6.829093 75:22.821554 76:6.340183 77:0 78:6.129836 79:0 80:10.145764 81:6.766068 82:0 83:6.820992 84:6.829093 85:12.67579 86:6.553125 87:0 88:6.475414 89:3.414546 90:11.410777 91:0.045344 92:0 93:0.119424 94:11.659127 95:1.600258 96:1 97:0 98:1 99:0 100:1 101:1 102:0 103:1 104:0.671329 105:0.989811 106:17.818264 107:0 108:10.183562 109:7.633816 110:19.436549 111:-6.340431 112:-12.071142 113:-7.191141 114:-13.131176 115:-5.755162 116:-13.631532 117:-16.095443 118:-14.367199 119:-16.975368 120:-12.63974 121:-5.692009 122:-12.91985 123:-5.00585 124:-13.980776 125:-5.509102 126:2 127:35 128:1 129:0 130:266 131:25070 132:28 133:7 134:0 135:0 136:0 "
//				.replaceAll("\\s[0-9|qid]*:", ";");
//		System.out.println(tmp);
	}

}
