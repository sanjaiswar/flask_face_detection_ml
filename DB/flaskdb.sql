-- phpMyAdmin SQL Dump
-- version 5.1.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Sep 02, 2022 at 08:28 AM
-- Server version: 10.4.22-MariaDB
-- PHP Version: 7.4.27

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `flaskdb`
--

-- --------------------------------------------------------

--
-- Table structure for table `employee`
--

CREATE TABLE `employee` (
  `id` int(11) NOT NULL,
  `name` varchar(100) DEFAULT NULL,
  `email` varchar(100) DEFAULT NULL,
  `phone` varchar(100) DEFAULT NULL,
  `pic` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `employee`
--

INSERT INTO `employee` (`id`, `name`, `email`, `phone`, `pic`) VALUES
(4, 'Santosh Singh', 'santosh@gmail.com', '8765002345', 'sanjaiswar.png'),
(5, 'Amit', 'amit@gmail.com', '1234567890', 'sanjaiswar.png'),
(6, 'Biresh', 'biresh@gmail.com', '1234567890', 'sanjaiswar.png'),
(7, 'Chetan', 'chetan@gmail.com', '1234567890', 'sanjaiswar.png'),
(8, 'Dileep', 'dileep@gmail.com', '1234567890', 'sanjaiswar.png'),
(9, 'Rupesh', 'rupesh@gmail.com', '1234567890', 'sanjaiswar.png'),
(10, 'Santosh', 'santosh@gmail.com', '1234567890', 'sanjaiswar.png'),
(11, 'Tilpesh', 'tipesh@gmail.com', '1234567890', 'sanjaiswar.png'),
(12, 'Umesh', 'umesh@gmail.com', '1234567890', 'sanjaiswar.png'),
(13, 'Praveen', 'praveen@gmail.com', '1234567890', 'sanjaiswar.png'),
(14, 'Rustom', 'runstom@gmail.com', '1234567890', 'sanjaiswar.png'),
(15, 'Suresh', 'suresh@gmail.com', '1234567890', 'sanjaiswar.png'),
(16, 'Durgesh', 'durgesh@gmail.com', '1234567890', 'sanjaiswar.png'),
(17, 'Sumit Chaudhary', 'sumit@yahoo.com', '7856453423', 'sanjaiswar.png');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `employee`
--
ALTER TABLE `employee`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `employee`
--
ALTER TABLE `employee`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=18;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
