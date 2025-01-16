import { useState } from "react";

export default function App() {

  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const [images, setImages] = useState([]);
  const [labels, setLabels] = useState([]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);

    // Read the file as a Data URL to show a preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result); // Set the preview image URL
    };
    reader.readAsDataURL(selectedFile);
  };

  const onSubmit = async () => {

    if (file) {
      // Create a FormData object and append the file
      const formData = new FormData();
      formData.append("file", file);

      try {
        // Send the file to the backend via a POST request
        const res = await fetch("http://127.0.0.1:3050/uploadImage", {
          method: "POST",
          body: formData,
        }).then(
          response => {
            if (response.ok) {
              return response.json()
            }
          }
        );
        console.log(res)
        let src = await res.images.map((image) => {
          return "data:image/jpeg;base64," + image
        });
        setImages(src);
        setLabels(res.label);
      } catch (error) {
        alert(error)
      }
    }

  }
  return (
    <div className=" min-h-screen bg-slate-500 flex flex-col items-center pt-16">
      <label htmlFor="myfile">Select a file:</label>
      <input className="mb-3" type="file" id="myfile" name="myfile" onChange={handleFileChange} />
      {file && (
        <img src={imagePreview} alt="#" />
      )}
      {file &&
        <button onClick={onSubmit} className="mt-5 px-2 py-1 rounded-md bg-white">Detect</button>
      }
      <div className="flex flex-row flex-wrap mx-16">
        {images.length > 0 &&
          images.map((e, i) => {
            return <div className="flex flex-col mx-5">
              <p className=" font-bold text-3xl">{labels[i]}</p>
              <img src={images[i]} alt="#" />
            </div>
          })
        }
      </div>
    </div>
  )
}