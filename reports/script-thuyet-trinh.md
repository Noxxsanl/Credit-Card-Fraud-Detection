# Script thuyết trình — Phát hiện gian lận thẻ tín dụng

Script nói theo đúng thứ tự 12 mục của [bao-cao.md](bao-cao.md), không dùng slide. Bài nói dài khoảng
**18 phút**. Demo làm **sau** bài nói, theo [ke-hoach-demo.md](ke-hoach-demo.md). Phần Q&A ở cuối file để
ôn, không đọc lên.

## Cách dùng

| Phần | Mục trong báo cáo | Thời gian |
|---|---|---|
| Mở đầu | Tóm tắt | 1 phút |
| 1. Đặt vấn đề và dữ liệu | §1 | 1,5 phút |
| 2. Khám phá dữ liệu | §2 | 1,5 phút |
| 3. Kiểm định thống kê và xếp hạng đặc trưng | §3 | 1 phút |
| 4. Khung đánh giá | §4 | 1,5 phút |
| 5. Mô hình cơ sở | §5 | 1 phút |
| 6. So sánh chiến lược xử lý mất cân bằng | §6 | 3 phút |
| 7. Mô hình cuối, tinh chỉnh và khoảng tin cậy | §7 | 2 phút |
| 8. Ngưỡng quyết định và chi phí | §8 | 3 phút |
| 9. Giải thích mô hình và phân tích lỗi | §9 | 1 phút |
| 10. Ứng dụng demo | §10 | 0,5 phút |
| 11. Hạn chế và hướng mở rộng | §11 | 1 phút |
| 12. Kết luận | §12 | 1 phút |

Ký hiệu:

- **[NHẤN]**: câu cần nói chậm, nhấn giọng — thường là con số hoặc kết luận chính.
- *(có thể bỏ)*: đoạn bỏ được khi thiếu giờ. Bỏ hết các đoạn này thì bài còn khoảng 14 phút; nội dung của
  chúng đã nằm trong phần Q&A.
- Đoạn chữ thường là lời nói, xưng "em" với thầy cô.

---

## Mở đầu (1 phút)

Em xin chào thầy cô và các bạn. Em là Nguyễn Hoàng Đạt. Hôm nay em trình bày đồ án "Phát hiện gian lận
thẻ tín dụng trên dữ liệu cực kỳ mất cân bằng".

Em xin bắt đầu bằng một câu hỏi. Nếu em nói mô hình của em có accuracy 99,83%, thầy cô có tin nó tốt không?

**[NHẤN]** Mô hình đó không học gì cả. Nó chỉ luôn trả lời "giao dịch hợp lệ". Lý do là cứ khoảng 600 giao
dịch mới có một vụ gian lận. Đó là cái khó của đề tài này: con số quen thuộc nhất lại đánh lừa mình nhiều nhất.

Em sẽ trình bày theo 12 mục của báo cáo, từ dữ liệu, cách đánh giá, so sánh các chiến lược, tới cách chọn
ngưỡng và hệ thống chạy được. Cuối bài em xin demo hệ thống.

---

## 1. Đặt vấn đề và dữ liệu (1,5 phút)

Bài toán là: với mỗi giao dịch thẻ, hệ thống trả về một điểm rủi ro từ 0 đến 1. Giao dịch có điểm vượt
ngưỡng sẽ vào hàng đợi để người thẩm định xem.

Hai loại lỗi ở đây có giá rất khác nhau. Bỏ lọt một vụ gian lận thì mất tiền của giao dịch đó. Còn một cảnh
báo giả chỉ tốn công thẩm định. **[NHẤN]** Vì vậy em tách bài toán làm hai phần: xếp hạng giao dịch theo rủi
ro là việc của mô hình; còn chọn ngưỡng là việc của nghiệp vụ, và có thể thay đổi lúc chạy.

Dữ liệu là bộ Credit Card Fraud Detection của Đại học Tự do Brussels, phát hành trên Kaggle, giấy phép DbCL.
Bộ này có 284.807 giao dịch của chủ thẻ châu Âu trong hai ngày, chỉ 492 vụ gian lận. Các cột V1 đến V28 là
thành phần PCA đã ẩn danh; ngoài ra có số tiền, thời gian và nhãn.

Em loại 1.081 dòng trùng lặp **trước** khi chia tập — nếu loại sau, cùng một dòng có thể nằm cả ở tập huấn
luyện lẫn tập kiểm thử. Còn lại 283.726 dòng với 473 vụ gian lận. Em chia phân tầng 80/20: tập kiểm thử có
56.746 giao dịch nhưng chỉ 95 vụ gian lận. **[NHẤN]** Tập này chỉ dùng để đo — không chọn mô hình, siêu tham
số hay ngưỡng nào trên nó.

Bộ dữ liệu không ghi hai điều, nên em phải quy ước: đơn vị tiền là euro, và giờ bắt đầu của cột Time là 0 giờ.
Em không đưa Time thô vào mô hình, vì số giây trong một cửa sổ hai ngày không tồn tại khi triển khai. Em chỉ
giữ giờ trong ngày, mã hoá sin–cos để 23 giờ và 0 giờ nằm cạnh nhau. Tổng cộng 31 đặc trưng, sinh bởi một hàm
dùng chung cho cả notebook lẫn API.

---

## 2. Khám phá dữ liệu (1,5 phút)

Em có năm phát hiện chính.

Thứ nhất, tỷ lệ 1 trên 600 loại accuracy khỏi cuộc chơi. Đường cơ sở của PR-AUC bằng đúng tỷ lệ gian lận,
0,00167.

Thứ hai, số tiền một mình không phân biệt được gian lận. Trung vị của gian lận là 9,82 euro, thấp hơn giao
dịch hợp lệ, nhưng trung bình lại cao hơn — đuôi dày ở cả hai phía. Đáng chú ý là 36% vụ gian lận có số tiền
dưới 1 euro. Chi tiết này sẽ quay lại ở mục 8.

Thứ ba, gian lận dồn vào giờ vắng. **[NHẤN]** Lúc 2 giờ sáng, tỷ lệ gian lận là 1,45%; lúc 10 giờ sáng chỉ
0,048% — chênh 30 lần. Đây là cơ sở để đưa giờ vào mô hình, và em đã kiểm lại điều này chỉ trên tập huấn
luyện.

Thứ tư, V1 đến V28 gần như trực giao với nhau, nên không cần chạy lại PCA.

Thứ năm, không đặc trưng đơn nào tách được hai lớp, nhưng bài toán khả thi. Trên hình chiếu t-SNE, 85,8% mẫu
gian lận tạo thành hai, ba cụm tách hẳn. Còn 21 mẫu, khoảng 4,4%, nằm lẫn hoàn toàn giữa giao dịch hợp lệ —
loại mô hình khó bắt nhất.

---

## 3. Kiểm định thống kê và xếp hạng đặc trưng (1 phút)

Em kiểm định từng đặc trưng bằng Mann–Whitney U, vì nó không giả định phân phối chuẩn. Em hiệu chỉnh
Benjamini–Hochberg cho 30 kiểm định đồng thời, và tính thêm độ lớn hiệu ứng — Cohen's d và Cliff's delta. Với
283.000 mẫu, p-value gần như luôn nhỏ; độ lớn hiệu ứng mới cho biết khác biệt có đáng kể hay không.

Kết quả: 27 trên 30 đặc trưng còn ý nghĩa sau hiệu chỉnh; ba đặc trưng không có ý nghĩa là V13, V15, V22.

Có hai điều đáng giữ. **[NHẤN]** Một, V17 có Cohen's d bằng −8 — hiệu ứng khổng lồ — nhưng tương quan
Pearson chỉ −0,31. Đó không phải mâu thuẫn, mà là hệ quả của tỷ lệ lớp 0,17%: độ lớn tương quan bị nén xuống,
cùng một cái bẫy như accuracy 99,83%.

Hai, hai thước đo hiệu ứng bất đồng về thứ hạng: V17 hạng 1 theo d nhưng hạng 11 theo Cliff's delta. Câu hỏi
"đặc trưng nào thật sự quan trọng" em trả lời lại bằng SHAP ở mục 9. Phân tích này chỉ để hiểu dữ liệu, không
dùng để chọn đặc trưng, nên không gây rò rỉ.

---

## 4. Khung đánh giá (1,5 phút)

Em dùng PR-AUC, không dùng accuracy hay ROC-AUC. ROC-AUC không đủ vì có hơn 56.000 giao dịch hợp lệ: thêm vài
trăm cảnh báo giả mà tỷ lệ dương giả gần như không nhúc nhích. PR-AUC đo đúng câu hỏi: trong những giao dịch bị
gắn cờ, bao nhiêu là gian lận thật.

Em đặt ra ba nguyên tắc.

Một, mọi chỉ số đều kèm khoảng tin cậy bootstrap phân tầng. Tập kiểm thử chỉ có 95 vụ gian lận, mỗi vụ bắt hụt
là recall giảm hơn 1 điểm phần trăm. So sánh hai mô hình thì em dùng bootstrap theo cặp, và với các cặp cần
kết luận thì dùng CV lặp kèm kiểm định t hiệu chỉnh Nadeau–Bengio.

Hai, một quy tắc đặt trước: nếu khoảng tin cậy của hiệu chứa 0, em **không** tuyên bố bên nào hơn.

Ba, chống rò rỉ dữ liệu. Bảy quy tắc được viết thành kiểm thử tự động — ví dụ SMOTE chỉ được nằm trong
pipeline, bộ chuẩn hoá chỉ fit trên dữ liệu huấn luyện của từng fold. *(có thể bỏ)* Em còn có kiểm thử theo
hành vi: xáo trộn nhãn thì PR-AUC phải về mức may rủi, và cố ý chạy SMOTE trước khi chia fold thì kiểm thử phải
bắt được.

---

## 5. Mô hình cơ sở (1 phút)

Em bắt đầu với hai mô hình đơn giản, chưa xử lý mất cân bằng, ngưỡng 0,5: Logistic Regression đạt PR-AUC
0,696, Decision Tree đạt 0,628.

**[NHẤN]** Accuracy xếp hạng ngược hai mô hình này: cây có accuracy cao hơn — 99,938% so với 99,915% — nhưng
PR-AUC thấp hơn. Ba mô hình, kể cả mô hình rỗng, chỉ chênh nhau 0,11 điểm phần trăm accuracy.

Em cũng chưa tuyên bố mô hình nào thắng: hiệu PR-AUC giữa hai mô hình có khoảng tin cậy chứa 0.

*(có thể bỏ)* Một chi tiết: cây sâu 6 tầng chỉ phát ra 8 giá trị điểm khác nhau, nên đường PR của nó là cầu
thang và không thể tối ưu ngưỡng trên nó. Đây là cảnh báo cho mục sau.

---

## 6. So sánh chiến lược xử lý mất cân bằng (3 phút)

Đây là trọng tâm của đồ án. Mục tiêu không phải tìm một mô hình thắng, mà hiểu mỗi kỹ thuật xử lý mất cân bằng
đổi được gì và mất gì.

Em so sánh năm chiến lược — không xử lý, trọng số lớp, undersampling, SMOTE và SMOTE + Tomek — trên bốn mô hình:
Logistic Regression, Decision Tree, Random Forest và XGBoost. Tổng cộng 20 tổ hợp, kiểm định chéo 5 fold trên
tập huấn luyện, mọi bước nằm trong pipeline. Sau đó em làm thêm hai thí nghiệm: đặt các chiến lược ở cùng mức
cân bằng — thêm 48 tổ hợp — và chạy CV lặp 5 lần 5 fold.

**[NHẤN]** Kết quả đầu tiên: mô hình quyết định nhiều hơn chiến lược. XGBoost và Random Forest đều cao, Decision
Tree thấp bất kể chiến lược.

Em có năm phát hiện.

Một, sáu cấu hình dẫn đầu hoà nhau, quanh PR-AUC 0,855. Khoảng cách giữa chúng chưa tới một phần tám độ lệch
chuẩn giữa các fold. Kiểm lại bằng CV lặp thì mọi khoảng tin cậy đều chứa 0. Nên em nói "một nhóm dẫn đầu",
không nói "tổ hợp X thắng".

Hai, trọng số lớp hay SMOTE, phần lớn là chuyện **mức** cân bằng chứ không phải kỹ thuật. Ban đầu Decision Tree
trông như sụp vì trọng số lớp. Nhưng đặt hai kỹ thuật ở cùng tỷ lệ 1 : 10 thì chúng bằng nhau — 0,6416 so với
0,6404. Thứ làm cây sụp là cân bằng hoàn toàn 1 : 1.

Ba, SMOTE + Tomek cho kết quả giống SMOTE tới từng bit, mà tốn khoảng 40% thời gian cả lưới. Lý do: chính SMOTE
đã xoá hết các cặp Tomek trước khi bước Tomek chạy. Em giữ kết quả âm tính này vì tự nó là một phát hiện.

Bốn, *(có thể bỏ)* Random Forest hoà với XGBoost nhưng chỉ có khoảng 200 mức điểm, so với khoảng 222.000 của
XGBoost — không đủ mịn để kéo ngưỡng.

Năm, *(có thể bỏ)* undersampling nhanh nhất nhưng kém nhất ở mọi mô hình và mọi mức cân bằng, vì mô hình thấy
quá ít giao dịch hợp lệ "khó".

Cuối cùng, em chọn cấu hình bằng một quy tắc đặt trước: trong các cấu hình XGBoost hoà nhau, chọn chiến lược đơn
giản nhất. Cả chín ứng viên đều hoà, nên kết quả là XGBoost **không xử lý mất cân bằng**.

**[NHẤN]** Đây là kết luận trung tâm: với mô hình boosting trên dữ liệu này, xử lý mất cân bằng không đem lại lợi
ích đo được. Trọng số lớp và SMOTE chỉ dịch điểm rủi ro, không đổi thứ tự xếp hạng — mà PR-AUC chỉ đo thứ tự.
Việc thật sự quyết định là chọn ngưỡng.

---

## 7. Mô hình cuối, tinh chỉnh và khoảng tin cậy (2 phút)

Em thử tinh chỉnh siêu tham số với 30 lần thử. Đây là một kết quả âm tính: cấu hình tốt nhất chỉ hơn cấu hình
mặc định 0,0034, khoảng tin cậy chứa 0. Theo quy tắc đặt trước, em giữ cấu hình mặc định.

Huấn luyện lại trên toàn bộ tập huấn luyện và đo một lần trên tập kiểm thử: **[NHẤN]** PR-AUC đạt 0,828, khoảng
tin cậy từ 0,747 đến 0,900 — gấp gần 500 lần đường cơ sở. Các chỉ số tại ngưỡng em sẽ trình bày ở mục 8.

Chia ngẫu nhiên thì trộn giao dịch của cả hai ngày vào cả hai tập. Để sát với triển khai hơn, em học trên ngày 1
và chấm ngày 2: PR-AUC còn 0,773. Em có một dòng đối chứng cho thấy phần giảm do thời gian là khoảng 0,044.

*(có thể bỏ)* Em cũng thử một autoencoder không dùng nhãn để đối chứng. Nó chỉ đạt PR-AUC 0,329. Bài toán có
nhãn thì học có giám sát thắng.

*(có thể bỏ)* Về tái lập: chạy lại toàn bộ 9 notebook từ đầu, PR-AUC lệch đúng 0.

---

## 8. Ngưỡng quyết định và chi phí (3 phút)

Ngưỡng 0,5 ngầm giả định hai loại lỗi tốn như nhau. Em thay bằng một hàm chi phí với ba giả định: bỏ lọt một vụ
mất khoảng 116 euro — trung bình số tiền gian lận trên tập huấn luyện —, thẩm định một cảnh báo tốn 5 euro, và
đội thẩm định xử lý tối đa 200 cảnh báo mỗi ngày. Cả ba chưa được nghiệp vụ xác nhận.

Ngưỡng là một phép chọn, nên em không chọn trên tập kiểm thử. Em chọn trên điểm forward-chaining: sắp tập huấn
luyện theo thời gian, và mỗi khối chỉ được chấm bởi mô hình học từ các khối trước nó — mô phỏng việc chấm tương
lai.

Em thử mọi giá trị điểm làm ngưỡng — hơn 175.000 ứng viên — và lấy ngưỡng rẻ nhất trong ngân sách 200 cảnh báo
mỗi ngày. Kết quả là ngưỡng mặc định **0,2675**. Nó đúng bằng điểm của vụ gian lận cuối cùng còn bắt được mà chưa
vượt ngân sách.

Trên tập kiểm thử, ngưỡng này bắt 72 trên 95 vụ, recall 0,758, precision 0,96, chỉ 3 cảnh báo giả, với 187
cảnh báo mỗi ngày.

**[NHẤN]** Em xin nói thẳng: tiêu chí recall 0,75 đạt theo ước lượng điểm, nhưng sát nút. Một vụ ít hơn là trượt,
và cận dưới khoảng tin cậy là 0,674. Đó là hệ quả của việc chỉ có 95 mẫu dương.

Nếu bỏ ràng buộc ngân sách, ngưỡng tối ưu chi phí là 0,0191. Nó cho recall 0,821 và rẻ hơn 616 euro trên tập
kiểm thử, nhưng cần khoảng 300 cảnh báo mỗi ngày — vượt ngân sách. **[NHẤN]** Vì vậy thứ quyết định ngưỡng là
năng lực thẩm định, không phải hàm chi phí.

Em kiểm độ nhạy của kết luận này. Ngưỡng 0,0191 không đổi khi tỷ lệ chi phí chạy từ 15 : 1 đến 75 : 1. Nhưng
nếu tính chi phí bỏ lọt bằng số tiền thật của từng vụ, lợi ích của việc hạ ngưỡng co lại hơn 4 lần và mất ý nghĩa
thống kê — vì 36% vụ gian lận dưới 1 euro. Dạng hàm chi phí là quyết định của nghiệp vụ.

Em cũng kiểm giao thức chọn ngưỡng trên một ngày thật sự mới: chỉ dùng ngày 1 để chọn, rồi áp cho ngày 2. Ngưỡng
trong ngân sách vẫn giữ được tải, nhưng recall chỉ còn khoảng 0,70. Nên ngưỡng không thể đặt một lần rồi để đó,
mà phải được theo dõi sau khi áp dụng.

*(có thể bỏ)* Hệ thống phân ba mức: cho qua, cần thẩm định, và đề xuất chặn. Ngưỡng chặn chọn theo precision ≥
95%, ra 0,9995 — chỉ các vụ gần như chắc chắn mới bị chặn tự động. Còn điểm rủi ro không hẳn là xác suất ở hai
đầu, nên giao diện gọi là "điểm rủi ro".

---

## 9. Giải thích mô hình và phân tích lỗi (1 phút)

Theo SHAP, năm đặc trưng V4, V14, V12, V10 và V11 chiếm 49% tầm ảnh hưởng. **[NHẤN]** Thú vị là V17 — hạng 1
theo Cohen's d ở mục 3 — lại xếp cuối cùng theo SHAP. Nó thừa thông tin khi mô hình đã có V14, V12, V10. Đây là
câu trả lời cho câu hỏi để ngỏ ở mục 3.

Về lỗi: 61% vụ bị bỏ lọt có cả năm đặc trưng chính nằm trong vùng của giao dịch hợp lệ. Tức là bỏ lọt là giới hạn
của đặc trưng, không phải của ngưỡng. Còn 3 cảnh báo giả thì mang chữ ký của gian lận, một trong số đó là giao
dịch có số tiền lớn nhất bộ dữ liệu.

Giới hạn của mọi diễn giải ở đây: V1 đến V28 là PCA ẩn danh. SHAP chỉ nói được "V14 đẩy điểm lên", không nói được
nguyên nhân nghiệp vụ.

---

## 10. Ứng dụng demo (0,5 phút)

Em đóng gói mô hình thành một hệ thống chấm điểm và thẩm định: giao diện web, API FastAPI và PostgreSQL, dựng bằng
một lệnh `docker compose up`. API sinh đặc trưng bằng đúng hàm dùng trong notebook, nên không có lệch giữa huấn
luyện và phục vụ. Ngưỡng là cấu hình của hệ thống, đổi được lúc chạy, và có thẻ theo dõi tỷ lệ cảnh báo và
precision sau khi áp dụng.

Hệ thống có bốn màn hình — hàng đợi, chi tiết giao dịch, cấu hình ngưỡng, hiệu năng — và 406 ca kiểm thử tự động.
Em xin demo trực tiếp ở cuối bài.

---

## 11. Hạn chế và hướng mở rộng (1 phút)

Em xin nêu bốn hạn chế chính.

Một, V1 đến V28 là PCA ẩn danh, nên không diễn giải được theo nghiệp vụ.

Hai, dữ liệu chỉ có hai ngày, và ngay giữa hai ngày đó đã có dịch chuyển.

Ba, chỉ 95 vụ gian lận trong tập kiểm thử, nên khoảng tin cậy rất rộng và nhiều so sánh không kết luận được.

Bốn, chi phí và ngân sách thẩm định là giả định, chưa được nghiệp vụ xác nhận.

Hướng tiếp theo: hiệu chỉnh điểm thành xác suất, cho phép chọn ngưỡng theo số tiền từng giao dịch, theo dõi dịch
chuyển tự động, và thêm đặc trưng hành vi theo thẻ khi có dữ liệu phù hợp.

---

## 12. Kết luận (1 phút)

Em xin kết lại bằng ba thông điệp.

Một, ở tỷ lệ 1 trên 600, accuracy vô nghĩa. Phải đo bằng PR-AUC kèm khoảng tin cậy.

Hai, mô hình quyết định nhiều hơn chiến lược xử lý mất cân bằng. Với XGBoost, không xử lý gì cũng tốt ngang mọi kỹ
thuật. Mô hình cuối đạt PR-AUC 0,828 trên tập kiểm thử.

**[NHẤN]** Ba, chọn ngưỡng mới là trọng tâm. Ngưỡng do năng lực thẩm định quyết định, phải chọn trên dữ liệu mô
phỏng tương lai, và phải được theo dõi sau khi áp dụng. Vì thế em tách ngưỡng khỏi mô hình, thành một cấu hình mà
người dùng thay đổi được và thấy ngay cái giá của mỗi lựa chọn.

Sau đây em xin demo hệ thống để thầy cô thấy rõ điều này.

*(Chuyển sang trình duyệt, làm theo [ke-hoach-demo.md](ke-hoach-demo.md). Lời cảm ơn và mời đặt câu hỏi nói ở cuối
demo.)*

---

## Chuẩn bị Q&A

Câu hỏi hội đồng dễ hỏi nhất, kèm số liệu và mục trong `bao-cao.md` để tra nhanh.

| Câu hỏi | Gợi ý trả lời | Mục |
|---|---|---|
| Recall 0,758 mà cận dưới 0,674, vậy có thật sự đạt tiêu chí 0,75 không? | Đạt theo ước lượng điểm, không đạt "chắc chắn". 72/95, một vụ ít hơn là trượt. Nguyên nhân là chỉ 95 mẫu dương; em báo cáo đầy đủ thay vì giấu đi. | 7.2, 8.3 |
| Cơ sở để ra ngưỡng 0,2675? | Thử mọi giá trị điểm (175.627 ngưỡng) trên điểm forward-chaining của tập huấn luyện; tính chi phí 115,94 × FN + 5 × FP và số cảnh báo/ngày; lấy ngưỡng rẻ nhất trong ngân sách 200 cảnh báo/ngày. 0,2675 là điểm của vụ gian lận cuối cùng còn bắt được trong ngân sách: bắt 189/249, 198 cảnh báo/ngày; vụ kế tiếp ở 0,2657 thì vượt 200. | 8.2, 8.8 |
| Vì sao không dùng τ* = 0,0191 rẻ hơn? | τ* cần khoảng 300 cảnh báo/ngày, vượt ngân sách 200. Mọi ngưỡng rẻ hơn đều vượt ngân sách, nên năng lực thẩm định quyết định. τ* vẫn hiện trên giao diện để người dùng thấy cái giá của ngân sách. | 8.2, 8.3 |
| Con số 116 euro và 5 euro lấy ở đâu? | 115,94 là trung bình số tiền gian lận trên tập huấn luyện; 5 euro và 200 cảnh báo/ngày là giả định. Đổi nguồn của c_FN thì cả hai ngưỡng không đổi; ngưỡng trong ngân sách giữ 0,2675 với mọi tỷ lệ chi phí từ 7,5 : 1 đến 100 : 1. | 8.1, 8.4, 8.5 |
| Vì sao không dùng SMOTE dù nó phổ biến? | Với XGBoost, chín cấu hình hoà nhau theo CV lặp. SMOTE chỉ dịch điểm, không đổi thứ tự. Quy tắc đặt trước chọn cái đơn giản nhất; bỏ SMOTE còn cho Brier tốt hơn và ít cảnh báo giả hơn. | 6.8 |
| SMOTE + Tomek giống SMOTE tới từng bit — có phải bug không? | Không. Trên fold 1, trước SMOTE có 21 cặp Tomek, sau SMOTE còn 0: điểm tổng hợp trở thành láng giềng gần nhất của mỗi vụ gian lận thật, nên bước Tomek không xoá gì. | 6.5 |
| Làm sao chắc là không có rò rỉ dữ liệu? | Bảy quy tắc thành kiểm thử tự động, cộng hai kiểm thử theo hành vi: xáo nhãn thì PR-AUC phải về mức may rủi, và cố ý SMOTE trước khi chia fold thì phải bị bắt. PR-AUC cao nhất 0,857, xa ngưỡng báo động 0,95. | 4.3 |
| Random Forest hoà PR-AUC với XGBoost, sao lại loại? | Chỉ 200–269 mức điểm trên 226.980 giao dịch, so với khoảng 222.000 của XGBoost. Kéo ngưỡng thì hoặc không đổi gì, hoặc nhảy hàng chục cảnh báo một lúc. Ngoài ra chậm hơn nhiều. | 6.6 |
| Sao không dùng cấu hình đã tinh chỉnh? | Hơn mặc định +0,0034 [−0,0028; +0,0097], khoảng tin cậy chứa 0. Quy tắc đặt trước là giữ mặc định; tập kiểm thử sau đó cũng không đảo lại quyết định. | 7.1 |
| Kiểm định thống kê ở mục 3 có dùng để chọn đặc trưng không? | Không. Mô hình dùng đủ 31 đặc trưng; phân tích chỉ để hiểu dữ liệu, nên tính trên toàn bộ dữ liệu không gây rò rỉ. | 3 |
| V14 quan trọng nghĩa là gì về nghiệp vụ? | Không nói được. V1–V28 là thành phần PCA ẩn danh; gán ý nghĩa nghiệp vụ cho chúng là sai sự thật. Giao diện có chú thích cố định nhắc điều này. | 9.2 |
| Điểm rủi ro có phải xác suất không? | Gần khi tính gộp (Brier tốt hơn khoảng 3 lần đoán hằng số), nhưng lệch ở hai đầu. Không ảnh hưởng kết quả dựa trên thứ hạng; cần isotonic nếu đọc như xác suất. Vì vậy giao diện gọi là "điểm rủi ro". | 8.7 |
| Triển khai thật, dữ liệu thay đổi thì sao? | Ngưỡng chọn trên điểm forward-chaining; hệ thống theo dõi tỷ lệ cảnh báo và precision theo khoảng 6 giờ, đánh dấu khoảng lệch. Chưa có cảnh báo tự động hay tự chọn lại ngưỡng — đó là hướng mở rộng. | 7.3, 10.2, 11.3 |
| Sao không dùng học sâu? | Em đã thử autoencoder không giám sát: PR-AUC 0,329 so với 0,828. Nó nhận ra gian lận là bất thường nhưng không phân biệt được với giao dịch hợp lệ hiếm gặp. Với dữ liệu bảng 31 cột, boosting là lựa chọn mạnh. | 7.4 |
| Nếu giả định Time bắt đầu lúc 00:00 sai thì sao? | Đặc trưng giờ chỉ lệch pha mà vẫn lặp chu kỳ 24 giờ, nên mô hình không bị ảnh hưởng; chỉ cách đọc "2 giờ sáng" phụ thuộc giả định. Bằng chứng: 1h–6h là sáu giờ vắng nhất. | 1.4 |
| Vì sao không dùng ROC-AUC? | Với hơn 56.000 giao dịch hợp lệ, hạ ngưỡng thêm 682 cảnh báo giả mà FPR chỉ tăng 1,2 điểm phần trăm, trong khi khối lượng thẩm định tăng 7 lần. ROC-AUC của 20 tổ hợp đều 0,87–0,98, còn PR-AUC trải từ 0,24 đến 0,85. | 4.1, 6.2 |
