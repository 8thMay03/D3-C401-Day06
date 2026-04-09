# User Flow + Prompt Cases — Chatbot Hỗ Trợ Tài Xế XanhSM
**Team D3 - C401 | Track: XanhSM**

---

## Tổng quan logic demo

Chatbot hoạt động theo flow: Tài xế nhắn câu hỏi → AI nhận diện intent → Tra cứu knowledge base nội bộ → Trả lời ngắn gọn + hiện nút action.

**10 nhóm intent:**
1. **Chính sách & quy định** — hoa hồng, thưởng, kỷ luật
2. **Thu nhập & thanh toán** — giải ngân, tra cứu số dư, khiếu nại
3. **Xử lý tình huống** — tai nạn, khách hàng khó, xe hỏng giữa ca
4. **Tìm trạm sạc** — vị trí, tình trạng, điều hướng
5. **Chính sách mua xe** — trả góp, ưu đãi, điều kiện
6. **Quy trình xử lý đơn & đơn hàng** — nhận đơn, hủy đơn, trạng thái
7. **Phản ánh & giải trình** — khiếu nại bị phạt, phản ánh khách hàng
8. **Chính sách phúc lợi** — bảo hiểm, nghỉ phép, hỗ trợ y tế
9. **Tài khoản ứng dụng** — đăng nhập, đổi thông tin, lỗi app

---

## Nhóm 1 — Chính sách & Quy định

### Case 1.1 — Hỏi chính sách thưởng (AI đúng, high confidence)

**Tài xế hỏi:**
> "Tuần này tao cần chạy bao nhiêu cuốc để được thưởng?"

**AI trả lời:**
> 🟢 **Thưởng tuần (áp dụng từ 01/04/2026):**
> Hoàn thành **50 cuốc** trong tuần (T2–CN) → thưởng **200.000đ**
> Hoàn thành **80 cuốc** → thưởng **400.000đ**
>
> ⏰ Tuần này bạn còn **3 ngày** để đạt mốc.
>
> ---
> 📌 *Thông tin từ chính sách nội bộ XanhSM, cập nhật 01/04/2026*

**Nút action:** `✅ Đúng rồi` | `❓ Chưa rõ` | `📞 Liên hệ CSKH`

---

### Case 1.2 — Hỏi quy định bị phạt (AI đúng, cần dẫn nguồn vì nhạy cảm)

**Tài xế hỏi:**
> "Hủy cuốc nhiều quá có bị khóa tài khoản không?"

**AI trả lời:**
> ⚠️ **Quy định hủy cuốc:**
> - Tỷ lệ hủy > **15%** trong 7 ngày → cảnh báo lần 1
> - Tỷ lệ hủy > **25%** → tạm khóa tài khoản 48 giờ
> - Vi phạm lần 3 trong 30 ngày → xem xét chấm dứt hợp tác
>
> 📌 *Đây là thông tin tham khảo. Quy định cụ thể có thể thay đổi — vui lòng xác nhận với CSKH nếu tài khoản đang bị ảnh hưởng.*

**Nút action:** `✅ Rõ rồi` | `🔄 Xem lại thông tin` | `📞 Liên hệ CSKH`

---

### Case 1.3 — AI không chắc (low confidence, câu hỏi mơ hồ)

**Tài xế hỏi:**
> "Chính sách mới nhất là gì?"

**AI trả lời:**
> 🤔 Câu hỏi của bạn khá rộng — mình chưa chắc bạn muốn hỏi về chủ đề nào.
> Bạn muốn xem thông tin về:

**Nút action:** `💰 Thưởng & hoa hồng` | `📋 Quy định vận hành` | `🚗 Bảo hiểm & tai nạn` | `📞 Hỏi CSKH`

> *(Tài xế chọn → AI hỏi tiếp theo nhánh tương ứng)*

---

## Nhóm 2 — Thu Nhập & Thanh Toán

### Case 2.1 — Tra cứu thu nhập hôm nay (real-time, AI đúng)

**Tài xế hỏi:**
> "Hôm nay tôi chạy được bao nhiêu tiền rồi?"

**AI trả lời:**
> 💵 **Thu nhập hôm nay (cập nhật lúc 14:32):**
>
> | Hạng mục | Số tiền |
> |---|---|
> | Cước chạy (12 cuốc) | 285.000đ |
> | Thưởng hoàn thành | 0đ |
> | Chiết khấu nền tảng | -57.000đ |
> | **Tổng thực nhận** | **228.000đ** |
>
> 📌 *Số liệu từ hệ thống Xanh Driver, có thể chênh lệch nhỏ do làm tròn.*

**Nút action:** `✅ OK` | `❓ Số này sai` | `📞 Khiếu nại thu nhập`

---

### Case 2.2 — Khiếu nại sai thu nhập (AI sai → recovery flow)

**Tài xế hỏi:**
> "Cuốc vừa rồi 45k sao app chỉ cộng 30k?"

**AI trả lời:**
> 🔍 Mình đang kiểm tra cuốc gần nhất của bạn...
>
> Theo hệ thống, cuốc **#XSM-20260408-1147** được ghi nhận: **30.000đ** (sau chiết khấu 33%).
>
> Nếu bạn nghĩ con số này không đúng, có thể do:
> - Cuốc áp dụng khuyến mãi giảm giá cho khách
> - Có phí điều chỉnh từ khiếu nại trước
>
> 📌 *Mình không thể xác nhận chắc chắn — vui lòng để CSKH kiểm tra lịch sử cuốc chi tiết.*

**Nút action:** `📞 Gửi khiếu nại ngay` | `🔄 Xem chi tiết cuốc` | `❓ Hỏi thêm`

---

### Case 2.3 — Hỏi ngày giải ngân (AI đúng)

**Tài xế hỏi:**
> "Tiền tuần trước giải ngân ngày nào?"

**AI trả lời:**
> 🏦 **Lịch giải ngân XanhSM:**
> Thu nhập từ **T2–CN** được giải ngân vào **thứ Tư tuần sau**, trước 17:00.
>
> Tuần trước (31/03–06/04) → giải ngân: **Thứ Tư 09/04/2026**
>
> 📌 *Nếu đến 18:00 thứ Tư chưa nhận được, hãy liên hệ CSKH kèm mã tài xế.*

**Nút action:** `✅ Hiểu rồi` | `📞 Chưa nhận được tiền`

---

## Nhóm 3 — Xử Lý Tình Huống

### Case 3.1 — Tai nạn nhỏ trong ca (AI đúng, high stakes)

**Tài xế hỏi:**
> "Tôi vừa va chạm nhẹ với xe máy, giờ phải làm gì?"

**AI trả lời:**
> 🚨 **Xử lý tai nạn — làm ngay theo thứ tự:**
>
> 1. **Đảm bảo an toàn** — dời xe vào lề nếu được, bật đèn hazard
> 2. **Kiểm tra người bị thương** — nếu có thương tích → gọi 113/115 ngay
> 3. **Chụp ảnh hiện trường** — xe, biển số, vết va chạm (ít nhất 4 góc)
> 4. **Không tự nhận lỗi** bằng lời trước khi có giám định
> 5. **Báo XanhSM ngay**: gọi hotline **1800 xxxx** hoặc nhấn nút bên dưới
>
> 📌 *Mình cung cấp hướng dẫn chung. Với tình huống phức tạp, hãy liên hệ CSKH ngay.*

**Nút action:** `📞 Gọi hotline ngay` | `📋 Xem quy trình đầy đủ` | `✅ Đã xử lý xong`

---

### Case 3.2 — Khách hàng khó, muốn hủy cuốc (AI đúng)

**Tài xế hỏi:**
> "Khách đang say xỉn, tôi không muốn chở, có được hủy không?"

**AI trả lời:**
> ✅ **Được phép hủy** trong trường hợp này, và **không bị tính vào tỷ lệ hủy** nếu báo đúng lý do.
>
> **Cách hủy đúng quy trình:**
> 1. Mở app → chọn cuốc hiện tại → "Báo cáo vấn đề"
> 2. Chọn lý do: **"Khách không đủ điều kiện đi xe"**
> 3. Mô tả ngắn (VD: khách có dấu hiệu say xỉn)
> 4. Xác nhận hủy → hệ thống ghi nhận miễn trừ
>
> 📌 *Nếu khách có hành vi đe dọa → gọi 113 và báo CSKH ngay.*

**Nút action:** `✅ Rõ rồi` | `📞 Cần hỗ trợ thêm`

---

### Case 3.3 — Xe hỏng giữa ca (AI không có đủ data → escalate)

**Tài xế hỏi:**
> "Xe xịt lốp giữa ca, tôi đang ở Đống Đa, cần làm gì?"

**AI trả lời:**
> 🔧 Mình chưa có thông tin đối tác sửa xe theo khu vực thời gian thực.
>
> **Làm ngay:**
> - Hủy cuốc hiện tại với lý do "Xe gặp sự cố kỹ thuật" (không tính vào tỷ lệ hủy)
> - Gọi **hotline kỹ thuật XanhSM: 1800 xxxx** để được hỗ trợ cứu hộ
>
> 📌 *CSKH có thể điều phối hỗ trợ tại chỗ hoặc tìm điểm vá lốp gần nhất.*

**Nút action:** `📞 Gọi hotline kỹ thuật` | `🗺️ Tìm tiệm sửa xe gần đây`

---

## Nhóm 4 — Tìm Trạm Sạc

### Case 4.1 — Tìm trạm sạc gần nhất (real-time, AI đúng)

**Tài xế hỏi:**
> "Trạm sạc gần tôi nhất ở đâu?"

**AI trả lời:**
> 🔋 **Trạm sạc gần bạn nhất (cập nhật lúc 15:10):**
>
> | Trạm | Khoảng cách | Cổng trống | Trạng thái |
> |---|---|---|---|
> | XanhSM – Hoàng Quốc Việt | 1,2 km | 4/8 cổng | 🟢 Hoạt động |
> | XanhSM – Cầu Giấy | 2,5 km | 1/6 cổng | 🟡 Sắp đầy |
> | XanhSM – Trần Duy Hưng | 3,1 km | 6/10 cổng | 🟢 Hoạt động |
>
> 📌 *Số cổng trống có thể thay đổi — kiểm tra lại khi đến gần.*

**Nút action:** `🗺️ Dẫn đường đến trạm 1` | `🔄 Cập nhật lại` | `📋 Xem tất cả trạm`

---

### Case 4.2 — Trạm sạc đầy / không hoạt động (AI đúng, có fallback)

**Tài xế hỏi:**
> "Trạm sạc Hoàng Quốc Việt có còn cổng không, tôi sắp đến rồi?"

**AI trả lời:**
> ⚠️ Trạm **XanhSM – Hoàng Quốc Việt** hiện đang **đầy** (0/8 cổng trống, cập nhật lúc 15:22).
>
> **Trạm thay thế gần đó:**
> - XanhSM – Cầu Giấy: **2 cổng trống**, cách đó 1,8 km
> - XanhSM – Nghĩa Đô: **5 cổng trống**, cách đó 2,2 km

**Nút action:** `🗺️ Dẫn đường đến Cầu Giấy` | `🗺️ Dẫn đường đến Nghĩa Đô` | `🔄 Kiểm tra lại`

---

### Case 4.3 — Hỏi giờ hoạt động / phí sạc (AI đúng)

**Tài xế hỏi:**
> "Trạm sạc hoạt động mấy giờ? Có mất phí không?"

**AI trả lời:**
> 🕐 **Giờ hoạt động:** 24/7 — tất cả trạm XanhSM
>
> 💳 **Phí sạc cho tài xế XanhSM:**
> - Sạc trong ca làm việc (đang có đơn active): **miễn phí**
> - Sạc ngoài ca: **1.500đ/kWh**, trừ thẳng vào ví tài xế
>
> 📌 *Cần đăng nhập app Xanh Driver để xác thực miễn phí sạc.*

**Nút action:** `✅ Rõ rồi` | `❓ Chưa rõ phí` | `📞 Liên hệ CSKH`

---

## Nhóm 5 — Chính Sách Mua Xe

### Case 5.1 — Hỏi điều kiện mua xe trả góp (AI đúng)

**Tài xế hỏi:**
> "Tôi mới chạy 2 tháng, có được mua xe trả góp qua XanhSM không?"

**AI trả lời:**
> 🚗 **Điều kiện mua xe trả góp dành cho tài xế XanhSM:**
>
> - Đã hợp tác tối thiểu **3 tháng liên tục** ✗ *(bạn đang ở tháng 2)*
> - Hoàn thành trung bình ≥ **30 cuốc/tuần** trong 2 tháng gần nhất
> - Không có vi phạm nghiêm trọng trong vòng 6 tháng
> - Điểm đánh giá trung bình ≥ **4.5 sao**
>
> ⏳ Bạn cần thêm khoảng **1 tháng** nữa để đủ điều kiện tháng hợp tác.
>
> 📌 *Điều kiện có thể thay đổi theo từng đợt. Liên hệ CSKH để được tư vấn cụ thể.*

**Nút action:** `📋 Xem chi tiết chương trình` | `🔔 Nhắc tôi khi đủ điều kiện` | `📞 Tư vấn CSKH`

---

### Case 5.2 — Hỏi ưu đãi mua xe, lãi suất (AI đúng, dẫn nguồn)

**Tài xế hỏi:**
> "Mua xe VF5 qua XanhSM thì lãi suất bao nhiêu, trả bao nhiêu năm?"

**AI trả lời:**
> 🏷️ **Chương trình mua xe VF 5 Plus (áp dụng Q2/2026):**
>
> | Hạng mục | Chi tiết |
> |---|---|
> | Giá xe | 458.000.000đ |
> | Trả trước tối thiểu | 30% (~137.400.000đ) |
> | Lãi suất ưu đãi | 6,9%/năm (12 tháng đầu) |
> | Thời hạn vay | 36 – 60 tháng |
> | Thưởng hợp tác | Khấu trừ tối đa 20tr vào đầu kỳ |
>
> 📌 *Đây là thông tin tham khảo — lãi suất thực tế phụ thuộc hồ sơ tín dụng. Vui lòng gặp tư vấn viên để có báo giá chính xác.*

**Nút action:** `📞 Đặt lịch tư vấn` | `📋 Xem các dòng xe khác` | `❓ Hỏi thêm`

---

### Case 5.3 — AI không đủ dữ liệu (low confidence → escalate)

**Tài xế hỏi:**
> "Nếu tôi nghỉ hợp tác giữa chừng thì xe trả góp tính sao?"

**AI trả lời:**
> 🤔 Đây là câu hỏi liên quan đến hợp đồng tài chính — mình không có đủ thông tin để trả lời chính xác cho trường hợp của bạn.
>
> Vui lòng liên hệ **bộ phận Mua xe XanhSM** để được giải thích điều khoản hợp đồng cụ thể.
>
> 📌 *Không nên tự suy đoán với vấn đề hợp đồng — tránh rủi ro quyền lợi.*

**Nút action:** `📞 Liên hệ bộ phận Mua xe` | `📋 Xem điều khoản hợp đồng mẫu`

---

## Nhóm 6 — Quy Trình Xử Lý Đơn & Đơn Hàng

### Case 6.1 — Hỏi cách nhận đơn / bị miss đơn (AI đúng)

**Tài xế hỏi:**
> "Tại sao tôi ít nhận được đơn hơn mấy anh khác cùng khu vực?"

**AI trả lời:**
> 📊 **Hệ thống phân bổ đơn dựa trên các yếu tố:**
>
> 1. **Điểm chấp nhận đơn** — tỷ lệ nhận đơn khi được gọi (nên > 80%)
> 2. **Điểm đánh giá** — sao trung bình 30 ngày gần nhất
> 3. **Khoảng cách** đến điểm đón khách
> 4. **Loại xe** — đơn VF8 chỉ phân bổ cho xe phù hợp
> 5. **Lịch sử hủy đơn** — tỷ lệ hủy cao sẽ giảm ưu tiên
>
> 💡 *Gợi ý: kiểm tra tỷ lệ chấp nhận đơn trong mục "Thống kê" trên app.*

**Nút action:** `📊 Xem thống kê của tôi` | `❓ Hỏi thêm` | `📞 Liên hệ CSKH`

---

### Case 6.2 — Đơn hàng bị lỗi / không hiện trên app (AI sai → recovery)

**Tài xế hỏi:**
> "Tôi vừa hoàn thành cuốc nhưng app không ghi nhận, mất đơn rồi?"

**AI trả lời:**
> 🔍 Mình đang kiểm tra lịch sử đơn gần nhất của bạn...
>
> Hệ thống **chưa ghi nhận** cuốc hoàn thành trong 30 phút vừa rồi. Có thể do:
> - Mạng bị gián đoạn khi bấm "Hoàn thành"
> - App bị lỗi cache
>
> **Làm ngay:**
> 1. Đóng app hoàn toàn → mở lại
> 2. Vào **Lịch sử chuyến** → kiểm tra lại
> 3. Nếu vẫn chưa hiện → báo CSKH kèm **thời gian** và **điểm đến** của cuốc
>
> 📌 *CSKH có thể tra cứu và cộng thủ công nếu xác nhận được cuốc.*

**Nút action:** `🔄 Kiểm tra lại lịch sử` | `📞 Báo CSKH ngay` | `✅ Đã hiện rồi`

---

### Case 6.3 — Hỏi trạng thái đơn đang chờ xử lý (AI đúng)

**Tài xế hỏi:**
> "Khiếu nại cuốc tôi gửi hôm qua xử lý chưa?"

**AI trả lời:**
> 🔎 **Trạng thái khiếu nại #KN-20260407-0892:**
>
> - Trạng thái: **Đang xử lý** 🟡
> - Gửi lúc: 14:05 ngày 07/04/2026
> - Bộ phận phụ trách: Đội kiểm soát chất lượng
> - Thời gian xử lý dự kiến: **1–3 ngày làm việc**
>
> 📌 *Kết quả sẽ được thông báo qua app và SMS.*

**Nút action:** `✅ OK, tôi chờ` | `📞 Cần xử lý gấp` | `🔄 Kiểm tra lại`

---

## Nhóm 7 — Phản Ánh & Giải Trình

### Case 7.1 — Phản ánh bị đánh giá sao thấp oan (AI đúng)

**Tài xế hỏi:**
> "Khách chấm tôi 1 sao nhưng tôi làm đúng hết rồi, tôi muốn khiếu nại"

**AI trả lời:**
> 📣 **Quy trình phản ánh đánh giá không công bằng:**
>
> 1. Vào app → **Lịch sử chuyến** → chọn chuyến bị đánh giá thấp
> 2. Nhấn **"Phản ánh đánh giá"**
> 3. Mô tả ngắn tình huống thực tế (VD: khách hủy nhưng vẫn chấm sao)
> 4. Đính kèm bằng chứng nếu có (ảnh chat, screenshot)
> 5. Gửi → nhận mã tra cứu
>
> ⏱️ Thời gian xem xét: **3–5 ngày làm việc**. Nếu được chấp thuận, đánh giá sẽ bị ẩn khỏi điểm trung bình.
>
> 📌 *Không phải mọi đánh giá thấp đều được xóa — chỉ những trường hợp có bằng chứng vi phạm chính sách review.*

**Nút action:** `📋 Phản ánh ngay trong app` | `📞 Liên hệ CSKH` | `❓ Hỏi thêm`

---

### Case 7.2 — Giải trình bị phạt vi phạm (AI đúng, high stakes)

**Tài xế hỏi:**
> "Tôi bị trừ điểm vì tội từ chối đón khách nhưng hôm đó xe tôi hỏng, tôi phải giải trình ở đâu?"

**AI trả lời:**
> 📝 **Quy trình giải trình vi phạm:**
>
> 1. Vào app → **Tài khoản** → **Vi phạm & Giải trình**
> 2. Chọn vi phạm cần giải trình (hiện ngày + loại vi phạm)
> 3. Viết giải trình: nêu rõ lý do khách quan, đính kèm bằng chứng
>    - *VD: ảnh xe hỏng, hóa đơn sửa xe, tin nhắn báo CSKH lúc đó*
> 4. Gửi → mã giải trình gửi về SMS
>
> ⏱️ Xem xét trong **5–7 ngày làm việc**. Nếu được chấp thuận, điểm vi phạm sẽ được hoàn trả.
>
> 📌 *Bằng chứng khách quan (ảnh, hóa đơn, lịch sử báo CSKH) tăng đáng kể khả năng giải trình thành công.*

**Nút action:** `📋 Mở mục Giải trình trong app` | `📞 Hỗ trợ soạn giải trình` | `❓ Hỏi thêm`

---

### Case 7.3 — Bị khách hàng phản ánh sai sự thật (AI đúng)

**Tài xế hỏi:**
> "Khách báo tôi chạy ẩu nhưng không đúng, tôi cần làm gì?"

**AI trả lời:**
> ⚠️ Phản ánh từ khách hàng đang ở trạng thái **đang xem xét** — chưa có quyết định chính thức.
>
> **Bạn có quyền phản hồi:**
> 1. Vào **Thông báo vi phạm** → chọn phản ánh liên quan
> 2. Nhấn **"Tôi muốn phản hồi"**
> 3. Mô tả sự việc theo góc nhìn của bạn + đính kèm bằng chứng:
>    - Camera hành trình (nếu có)
>    - Lịch sử GPS của chuyến
>    - Screenshot chat với khách
>
> 📌 *Dữ liệu GPS và camera từ hệ thống XanhSM cũng được đội kiểm soát truy xuất độc lập.*

**Nút action:** `📋 Phản hồi ngay` | `📞 Liên hệ CSKH để hỗ trợ` | `❓ Hỏi thêm`

---

## Nhóm 8 — Chính Sách Phúc Lợi

### Case 8.1 — Hỏi bảo hiểm tai nạn (AI đúng)

**Tài xế hỏi:**
> "Tôi có được bảo hiểm tai nạn không? Mức bồi thường bao nhiêu?"

**AI trả lời:**
> 🛡️ **Bảo hiểm tai nạn cho tài xế XanhSM (áp dụng trong ca làm việc):**
>
> | Loại | Mức bồi thường |
> |---|---|
> | Tử vong / thương tật vĩnh viễn | Tối đa **100.000.000đ** |
> | Thương tích phải nhập viện | Tối đa **20.000.000đ** |
> | Hỗ trợ chi phí y tế | Tối đa **5.000.000đ**/vụ |
>
> ✅ Bảo hiểm tự động kích hoạt khi bạn **đang có đơn active** trên app.
>
> 📌 *Tai nạn ngoài ca làm việc không thuộc phạm vi bảo hiểm này. Liên hệ CSKH để được hỗ trợ làm hồ sơ.*

**Nút action:** `📋 Xem điều khoản đầy đủ` | `📞 Báo tai nạn & làm hồ sơ` | `❓ Hỏi thêm`

---

### Case 8.2 — Hỏi chính sách nghỉ / hỗ trợ ốm đau (AI đúng)

**Tài xế hỏi:**
> "Tôi ốm không chạy được vài ngày có bị ảnh hưởng gì không?"

**AI trả lời:**
> 🏥 **Chính sách nghỉ do ốm đau / sức khỏe:**
>
> - Nghỉ ≤ **7 ngày/tháng**: không ảnh hưởng đến xếp hạng và phân bổ đơn
> - Nghỉ > 7 ngày: cần **thông báo trước** qua app → mục "Đăng ký nghỉ"
> - Có giấy xác nhận y tế: được miễn trừ ảnh hưởng đến điểm vi phạm
>
> 💊 **Hỗ trợ sức khỏe (tài xế chạy ≥ 6 tháng):**
> - Khám sức khỏe định kỳ miễn phí 1 lần/năm tại cơ sở đối tác
>
> 📌 *Đây là tài xế hợp tác, không phải nhân viên chính thức — chế độ phúc lợi có thể khác theo hợp đồng.*

**Nút action:** `📋 Đăng ký nghỉ trong app` | `📞 Liên hệ CSKH` | `❓ Hỏi thêm`

---

### Case 8.3 — Hỏi các phúc lợi khác (AI không chắc → hỏi làm rõ)

**Tài xế hỏi:**
> "XanhSM có hỗ trợ gì khác cho tài xế không?"

**AI trả lời:**
> 🎁 Mình có thể giới thiệu phúc lợi theo từng nhóm. Bạn quan tâm đến:

**Nút action:** `🛡️ Bảo hiểm & y tế` | `🎓 Đào tạo & nâng cấp bằng` | `🏷️ Ưu đãi đối tác (xăng, ăn uống)` | `🚗 Mua xe & vay vốn`

> *(Tài xế chọn → AI trả lời đúng nhóm)*

---

## Nhóm 9 — Tài Khoản Ứng Dụng

### Case 9.1 — Không đăng nhập được vào app (AI đúng, hướng dẫn từng bước)

**Tài xế hỏi:**
> "Tôi không đăng nhập được vào app Xanh Driver, báo sai mật khẩu"

**AI trả lời:**
> 🔐 **Thử lần lượt các bước sau:**
>
> 1. Kiểm tra số điện thoại đăng ký (đúng số SIM đang dùng không?)
> 2. Nhấn **"Quên mật khẩu"** → nhận OTP về SIM → đặt mật khẩu mới
> 3. Nếu không nhận được OTP: kiểm tra sóng, thử lại sau 2 phút
> 4. Vẫn không được → tài khoản có thể bị khóa tạm thời
>
> 📌 *Nếu số SIM đã thay đổi, bạn cần liên hệ CSKH để cập nhật — không thể tự đổi SIM qua app.*

**Nút action:** `🔑 Hướng dẫn đặt lại mật khẩu` | `📞 Liên hệ CSKH` | `✅ Đăng nhập được rồi`

---

### Case 9.2 — Muốn đổi thông tin tài khoản (AI đúng)

**Tài xế hỏi:**
> "Tôi muốn đổi số tài khoản ngân hàng nhận tiền, làm thế nào?"

**AI trả lời:**
> 🏦 **Cập nhật tài khoản ngân hàng:**
>
> Vì liên quan đến giải ngân, thay đổi này **bắt buộc xác minh danh tính**:
>
> 1. Vào app → **Tài khoản** → **Thông tin thanh toán** → **Đổi tài khoản ngân hàng**
> 2. Chụp ảnh CCCD (2 mặt) + selfie kèm CCCD
> 3. Nhập số tài khoản mới + tên ngân hàng
> 4. Xác nhận OTP về SIM
> 5. Đội kiểm duyệt xử lý trong **1–2 ngày làm việc**
>
> ⚠️ Trong thời gian xử lý, giải ngân vẫn về tài khoản cũ.
>
> 📌 *Chỉ đổi được 1 lần/3 tháng. Nếu cần gấp, liên hệ CSKH.*

**Nút action:** `📋 Mở mục đổi tài khoản` | `📞 Liên hệ CSKH` | `❓ Hỏi thêm`

---

### Case 9.3 — App bị lỗi / crash giữa ca (AI đúng, khẩn cấp)

**Tài xế hỏi:**
> "App bị crash, tôi đang có khách trên xe mà không bấm hoàn thành được"

**AI trả lời:**
> 🚨 **Xử lý ngay — app crash giữa cuốc:**
>
> 1. **Không cần hoảng loạn** — hệ thống vẫn ghi nhận GPS hành trình
> 2. Đưa khách đến đúng điểm đến như bình thường
> 3. Khởi động lại app → vào **Lịch sử** → cuốc vẫn hiện ở đó → bấm **"Hoàn thành"**
> 4. Nếu cuốc không hiện: gọi **hotline kỹ thuật 1800 xxxx** — đội hỗ trợ có thể hoàn thành từ hệ thống
>
> 📌 *Cuốc không bị mất — dữ liệu GPS lưu phía server, không phụ thuộc vào app.*

**Nút action:** `📞 Gọi hotline kỹ thuật ngay` | `✅ App hoạt động lại rồi` | `❓ Vẫn còn vấn đề`

---

## Logic demo — Flow tổng quát

```
Tài xế nhập câu hỏi
        │
        ▼
[Intent classifier]
        │
   ┌────┴────────────────────┐
   │                         │
High confidence          Low confidence / mơ hồ
   │                         │
   ▼                         ▼
Trả lời trực tiếp     Hỏi làm rõ (nút chọn)
+ dẫn nguồn                  │
   │                         ▼
   │                  Tài xế chọn nhánh
   │                         │
   └────────┬────────────────┘
            │
            ▼
      Hiện câu trả lời
      + nút feedback
            │
     ┌──────┴──────┐
     │             │
  ✅ Đúng      ❌ Sai / Chưa rõ
     │             │
  Kết thúc    → Xem lại / Escalate CSKH
                   │
              [Correction signal
               ghi vào hệ thống]
```

---

## Ghi chú cho phần demo

- **Dữ liệu giả lập:** thu nhập, số cuốc, ngày giải ngân dùng mock data
- **Intent chính cần demo được:** chính sách thưởng, hủy cuốc, tai nạn, tra cứu thu nhập, tìm trạm sạc, giải trình vi phạm, lỗi app
- **Luôn có nút "Liên hệ CSKH"** trong mọi response — đây là trust anchor chính
- **Câu trả lời tối đa 5 dòng** — nếu dài hơn thì chia nút "Xem thêm"

---

*User flow + Prompt Cases — Team D3 C401 — XanhSM Track — AI Thực Chiến · 2026*
