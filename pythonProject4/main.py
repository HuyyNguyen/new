import cv2
import face_recognition
imgMikami = face_recognition.load_image_file("pic\mikami.jpg") # Load hình ảnh mikami.jpg bằng thư viện face.recognition
imgMikami = cv2.cvtColor(imgMikami,cv2.COLOR_BGR2RGB)#Converts ảnh mikami.jpg từ BGR sang RGB

imgMikamicheck = face_recognition.load_image_file("mikamicheck.jpg")# Load hình ảnh mikamicheck.jpg bằng thư viện face.recognition
imgMikamicheck = cv2.cvtColor(imgMikamicheck,cv2.COLOR_BGR2RGB)#Converts ảnh mikamicheck.jpg từ BGR sang RGB
#Xac dinh khuon mat
faceLoc = face_recognition.face_locations(imgMikami)[0]# đưa ra 4 toạ độ giới hạn để nhận diện khuôn mặt
print(faceLoc) #(y1,x2,y2,x1)

#ma hoa hinh anh imgmikami
encodeMikami= face_recognition.face_encodings(imgMikami)[0]#Mã hoá hình ảnh imgMikami
cv2.rectangle(imgMikami,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)# Vẽ hình chữ nhật sự dụng cv2.rec.. đẩy về toạ độ x,y

#ma hoa hinh anh imgmikamicheck
faceCheck = face_recognition.face_locations(imgMikamicheck)[0]
encodeMikamicheck= face_recognition.face_encodings(imgMikamicheck)[0]#Mã hoá hình ảnh imgMikamicheck và được lưu lại
cv2.rectangle(imgMikamicheck,(faceCheck[3],faceCheck[0]),(faceCheck[1],faceCheck[2]),(255,0,255),2)# Vẽ hình chữ nhật sự dụng cv2.rec.. đẩy về toạ độ x,y


# so sánh mã hoã khuôn mặt 2 bức ảnh và trả về 2 kết quả T/F
ketqua= face_recognition.compare_faces([encodeMikami],encodeMikamicheck)
print(ketqua)

# So sánh mã hoã của 2 khuôn mặt và lấy khoảng cách euclidean cho biết mức độ giống nhau
distance= face_recognition.face_distance([encodeMikami],encodeMikamicheck)
print(ketqua,distance)

cv2.putText(imgMikamicheck,f"{ketqua}{round(distance[0],2)}",(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)

cv2.imshow("Mikami",imgMikami)
cv2.imshow("Mikamicheck",imgMikamicheck)
cv2.waitKey(0)